import os
import shutil
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Body,
    BackgroundTasks,
    Request,
    File,
    UploadFile,
)
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import torch
from transformers import pipeline
from .diarization_pipeline import diarize
from .utils import write_result, chunk2segment
import requests
import asyncio
import uuid
import pdb
from io import StringIO


admin_key = os.environ.get(
    "ADMIN_KEY",
)

hf_token = os.environ.get(
    "HF_TOKEN",
)

model_name = os.environ.get(
    "MODEL",
    "openai/whisper-large-v3"
)

# fly runtime env https://fly.io/docs/machines/runtime-environment
fly_machine_id = os.environ.get(
    "FLY_MACHINE_ID",
)

UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs=({"attn_implementation": "flash_attention_2"}),
)

app = FastAPI()
loop = asyncio.get_event_loop()
running_tasks = {}


class WebhookBody(BaseModel):
    url: str
    header: dict[str, str] = {}


def process(
    url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    diarise_audio: bool,
    formats: list[str] = None,  # New parameter
    webhook: WebhookBody | None = None,
    task_id: str | None = None,
):
    errorMessage: str | None = None
    outputs = {}
    try:
        generate_kwargs = {
            "task": task,
            "language": None if language == "None" else language,
        }

        outputs = pipe(
            url,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if timestamp == "word" else True,
        )

        if diarise_audio is True:
            speakers_transcript = diarize(
                hf_token,
                url,
                outputs,
            )
            outputs["speakers"] = speakers_transcript

        # New section to handle multiple formats
        if formats:
            outputs.update(format_outputs(outputs, formats))

    except asyncio.CancelledError:
        errorMessage = "Task Cancelled"
    except Exception as e:
        errorMessage = str(e)

    if task_id is not None:
        del running_tasks[task_id]

    if webhook is not None:
        webhookResp = (
            {"output": outputs, "status": "completed", "task_id": task_id}
            if errorMessage is None
            else {"error": errorMessage, "status": "error", "task_id": task_id}
        )

        if fly_machine_id is not None:
            webhookResp["fly_machine_id"] = fly_machine_id

        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(webhookResp),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs

def format_outputs(outputs, formats):
    formated = {}
    valid_formats = ['srt', 'vtt', 'lrc', 'tsv', 'txt']

    for format_type in formats:
        if format_type in valid_formats:
            # Create a StringIO buffer to write the formatted output
            buffer = StringIO()
            segments = {'segments': [chunk2segment(c) for c in outputs['chunks']]}
            write_result(segments, buffer, format_type)
            
            # Get the contents of the StringIO buffer
            formated[format_type] = buffer.getvalue()

            # Close the buffer
            buffer.close()

    return formated


@app.middleware("http")
async def admin_key_auth_check(request: Request, call_next):
    if admin_key is not None:
        if ("x-admin-api-key" not in request.headers) or (
            request.headers["x-admin-api-key"] != admin_key
        ):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response


@app.post("/")
def root(
    url: str = Body(),
    task: str = Body(default="transcribe", enum=["transcribe", "translate"]),
    language: str = Body(default="None"),
    batch_size: int = Body(default=64),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
    diarise_audio: bool = Body(default=False),
    formats: list[str] | None = Body(default=None, enum=['srt', 'vtt', 'lrc', 'tsv', 'txt']),
    webhook: WebhookBody | None = None,
    is_async: bool = Body(default=False),
    managed_task_id: str | None = Body(default=None),
):
    if url.lower().startswith("http") is False:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if diarise_audio is True and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if is_async is True and webhook is None:
        raise HTTPException(
            status_code=400, detail="Webhook is required for async tasks"
        )

    task_id = managed_task_id if managed_task_id is not None else str(uuid.uuid4())

    try:
        resp = {}
        if is_async is True:
            backgroundTask = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    process,
                    url,
                    task,
                    language,
                    batch_size,
                    timestamp,
                    diarise_audio,
                    formats,
                    webhook,
                    task_id,
                )
            )
            running_tasks[task_id] = backgroundTask
            resp = {
                "detail": "Task is being processed in the background",
                "status": "processing",
                "task_id": task_id,
            }
        else:
            running_tasks[task_id] = None
            outputs = process(
                url,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                formats,
                webhook,
                task_id,
            )
            resp = {
                "output": outputs,
                "status": "completed",
                "task_id": task_id,
            }
        if fly_machine_id is not None:
            resp["fly_machine_id"] = fly_machine_id
        return resp
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {"tasks": list(running_tasks.keys())}


@app.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]

    if task is None:
        return {"status": "processing"}
    elif task.done() is False:
        return {"status": "processing"}
    else:
        return {"status": "completed", "output": task.result()}


@app.delete("/cancel/{task_id}")
def cancel(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]
    if task is None:
        return HTTPException(status_code=400, detail="Not a background task")
    elif task.done() is False:
        task.cancel()
        del running_tasks[task_id]
        return {"status": "cancelled"}
    else:
        return {"status": "completed", "output": task.result()}


@app.post("/files")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the server
    
    Args:
    - file: The file to be uploaded
    
    Returns:
    - A dictionary with file details and upload status
    """
    # Generate a unique filename to prevent overwriting
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)

    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": unique_filename,
        "original_filename": file.filename,
        "status": "uploaded"
    }


@app.get("/files")
def list_files():
    """
    List all uploaded files
    
    Returns:
    - A list of filenames in the upload directory
    """
    files = os.listdir(UPLOAD_DIRECTORY)
    return {"files": files}


@app.get("/files/{filename}")
def download_file(filename: str):
    """
    Download a specific file
    
    Args:
    - filename: The name of the file to download
    
    Returns:
    - The file as a download
    """
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path, 
        media_type='application/octet-stream', 
        filename=filename
    )


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """
    Delete a specific file
    
    Args:
    - filename: The name of the file to delete
    
    Returns:
    - A status message confirming deletion
    """
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    os.remove(file_path)
    
    return {
        "filename": filename,
        "status": "deleted"
    }
