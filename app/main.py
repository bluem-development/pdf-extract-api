import time
from urllib import parse
import requests
from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException, Body
from celery.result import AsyncResult
from storage_manager import StorageManager
from celery_config import celery
from tasks import ocr_task, OCR_STRATEGIES
from hashlib import md5
import redis
import os
from pydantic import BaseModel, Field, field_validator
import ollama
import base64
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware


def storage_profile_exists(profile_name: str) -> bool:
    profile_path = os.path.abspath(os.path.join(os.getenv('STORAGE_PROFILE_PATH', '/storage_profiles'), f'{profile_name}.yaml'))
    return os.path.isfile(profile_path)

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Redis
redis_url = os.getenv('REDIS_CACHE_URL', 'redis://redis:6379/1')
redis_client = redis.StrictRedis.from_url(redis_url)

@app.post("/ocr")
async def ocr_endpoint(
    strategy: str = Form(...),
    model: str = Form(...),
    file: UploadFile = File(...),
    ocr_cache: bool = Form(...),
    prompt: str = Form(None),
    storage_profile: str = Form('default'),
    storage_filename: str = Form(None)
):
    """
    Endpoint to extract text from an uploaded PDF file using different OCR strategies.
    """
    # Convert string to boolean if needed
    if isinstance(ocr_cache, str):
        ocr_cache = ocr_cache.lower() == 'true'

    # Validate file type
    if file.content_type not in ['application/pdf', 'application/octet-stream']:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")

    pdf_bytes = await file.read()

    # Generate a hash of the PDF content for caching
    pdf_hash = md5(pdf_bytes).hexdigest()

    print(f"Processing PDF {file.filename} with strategy: {strategy}, ocr_cache: {ocr_cache}, model: {model}, storage_profile: {storage_profile}, storage_filename: {storage_filename}")

    # Asynchronous processing using Celery
    task = ocr_task.apply_async(args=[pdf_bytes, strategy, file.filename, pdf_hash, ocr_cache, prompt, model, storage_profile, storage_filename])
    return {"task_id": task.id}

# this is an alias for /ocr - to keep the backward compatibility
@app.post("/ocr/upload")
async def ocr_upload_endpoint(
    strategy: str = Form(...),
    model: str = Form(...),
    file: UploadFile = File(...),
    ocr_cache: bool = Form(...),
    prompt: str = Form(None),
    storage_profile: str = Form('default'),
    storage_filename: str = Form(None)
):
    """
    Alias endpoint to extract text from an uploaded PDF file using different OCR strategies.
    Supports both synchronous and asynchronous processing.
    """
    return await ocr_endpoint(
        strategy=strategy,
        model=model,
        file=file,
        ocr_cache=ocr_cache,
        prompt=prompt,
        storage_profile=storage_profile,
        storage_filename=storage_filename
    )

class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str

class OllamaPullRequest(BaseModel):
    model: str

class OcrRequest(BaseModel):
    strategy: str = Field(..., description="OCR strategy to use")
    prompt: Optional[str] = Field(None, description="Prompt for the Ollama model")
    model: str = Field(..., description="Model to use for the Ollama endpoint")
    file: str = Field(..., description="Base64 encoded PDF file")
    ocr_cache: bool = Field(..., description="Enable OCR result caching")
    storage_profile: Optional[str] = Field('default', description="Storage profile to use")
    storage_filename: Optional[str] = Field(None, description="Storage filename to use")

    @field_validator('strategy')
    def validate_strategy(cls, v):
        if v not in OCR_STRATEGIES:
            raise ValueError(f"Unknown strategy '{v}'. Available: marker, tesseract")
        return v

    @field_validator('file')
    def validate_file(cls, v):
        try:
            file_content = base64.b64decode(v)
            if not file_content.startswith(b'%PDF'):
                raise ValueError("Invalid file type. Only PDFs are supported.")
        except Exception:
            raise ValueError("Invalid file content. Must be base64 encoded PDF.")
        return v

    @field_validator('storage_profile')
    def validate_storage_profile(cls, v):
        if not storage_profile_exists(v):
            raise ValueError(f"Storage profile '{v}' does not exist.")
        return v

class OcrFormRequest(BaseModel):
    strategy: str = Field(..., description="OCR strategy to use")
    prompt: Optional[str] = Field(None, description="Prompt for the Ollama model")
    model: str = Field(..., description="Model to use for the Ollama endpoint")
    ocr_cache: bool = Field(..., description="Enable OCR result caching")
    storage_profile: Optional[str] = Field('default', description="Storage profile to use")
    storage_filename: Optional[str] = Field(None, description="Storage filename to use")

    @field_validator('strategy')
    def validate_strategy(cls, v):
        if v not in OCR_STRATEGIES:
            raise ValueError(f"Unknown strategy '{v}'. Available: marker, tesseract")
        return v

    @field_validator('storage_profile')
    def validate_storage_profile(cls, v):
        if not storage_profile_exists(v):
            raise ValueError(f"Storage profile '{v}' does not exist.")
        return v

@app.post("/ocr/request")
async def ocr_request_endpoint(request: OcrRequest):
    """
    Endpoint to extract text from an uploaded PDF file using different OCR strategies.
    Supports both synchronous and asynchronous processing.
    """
    # Validate input
    request_data = request.model_dump()
    try:
        print(request_data)
        OcrRequest(**request_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    file_content = base64.b64decode(request.file)

    # Process the file content as needed
    pdf_hash = md5(file_content).hexdigest()

    print(f"Processing PDF with strategy: {request.strategy}, ocr_cache: {request.ocr_cache}, model: {request.model}, storage_profile: {request.storage_profile}, storage_filename: {request.storage_filename}")

    # Asynchronous processing using Celery
    task = ocr_task.apply_async(args=[file_content, request.strategy, "uploaded_file.pdf", pdf_hash, request.ocr_cache, request.prompt, request.model, request.storage_profile, request.storage_filename])
    return {"task_id": task.id}

@app.get("/ocr/result/{task_id}")
async def ocr_status(task_id: str):
    """
    Endpoint to get the status of an OCR task using task_id.
    """
    task = AsyncResult(task_id, app=celery)

    if task.state == 'PENDING':
        return {"state": task.state, "status": "Task is pending..."}
    elif task.state == 'PROGRESS':
        task_info = task.info
        if task_info.get('start_time'):
            task_info['elapsed_time'] = time.time() - int(task_info.get('start_time'))
        return {"state": task.state, "status": task.info.get("status"), "info": task_info } 
    elif task.state == 'SUCCESS':
        return {"state": task.state, "status": "Task completed successfully.", "result": task.result}
    else:
        return {"state": task.state, "status": str(task.info)}

@app.post("/ocr/clear_cache")
async def clear_ocr_cache():
    """
    Endpoint to clear the OCR result cache in Redis.
    """
    redis_client.flushdb()
    return {"status": "OCR cache cleared"}

@app.get("/storage/list")
async def list_files(storage_profile: str = 'default'):
    """
    Endpoint to list files using the selected storage profile.
    """
    storage_manager = StorageManager(storage_profile)
    files = storage_manager.list()
    return {"files": files}

@app.get("/storage/load")
async def load_file(file_name: str, storage_profile: str = 'default'):
    """
    Endpoint to load a file using the selected storage profile.
    """
    storage_manager = StorageManager(storage_profile)
    content = storage_manager.load(file_name)
    return {"content": content}

@app.delete("/storage/delete")
async def delete_file(file_name: str, storage_profile: str = 'default'):
    """
    Endpoint to delete a file using the selected storage profile.
    """
    storage_manager = StorageManager(storage_profile)
    storage_manager.delete(file_name)
    return {"status": f"File {file_name} deleted successfully"}

@app.post("/llm/pull")
async def pull_llama(request: OllamaPullRequest):
    """
    Endpoint to pull the latest Llama model from the Ollama API.
    """
    print("Pulling " + request.model)
    try:
        response = ollama.pull(request.model)
    except ollama.ResponseError as e:
        print('Error:', e.error)
        raise HTTPException(status_code=500, detail="Failed to pull Llama model from Ollama API")

    return {"status": response.get("status", "Model pulled successfully")}

@app.post("/llm/generate")
async def generate_llama(request: OllamaGenerateRequest):
    """
    Endpoint to generate text using Llama 3.1 model (and other models) via the Ollama API.
    """
    print(request)
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    try:
        response = ollama.generate(request.model, request.prompt)
    except ollama.ResponseError as e:
        print('Error:', e.error)
        if e.status_code == 404:
            print("Error: ", e.error)
            ollama.pull(request.model)

        raise HTTPException(status_code=500, detail="Failed to generate text with Ollama API")

    generated_text = response.get("response", "")
    return {"generated_text": generated_text}
