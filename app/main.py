import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from celery.result import AsyncResult
from celery_config import celery
from tasks import ocr_task, OCR_STRATEGIES
from hashlib import md5
import redis
import os

app = FastAPI()

# Connect to Redis
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
redis_client = redis.StrictRedis.from_url(redis_url)

OLLAMA_API_URL = "http://ollama:8000/generate"  # URL to call the Ollama API within Docker

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...), strategy: str = "marker", async_mode: bool = True, ocr_cache: bool = False):
    """
    Endpoint to extract text from an uploaded PDF file using different OCR strategies.
    Supports both synchronous and asynchronous processing.
    """
    if file.content_type is not None and file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")

    pdf_bytes = await file.read()

    # Generate a hash of the PDF content for caching
    pdf_hash = md5(pdf_bytes).hexdigest()

    if ocr_cache:
        cached_result = redis_client.get(pdf_hash)
        if cached_result:
            # Return cached result if available
            return {"text": cached_result.decode('utf-8')}

    if strategy not in OCR_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{strategy}'. Available: marker, tesseract")

    if async_mode:
        # Asynchronous processing using Celery
        task = ocr_task.apply_async(args=[pdf_bytes, strategy])
        return {"task_id": task.id}
    else:
        # Synchronous processing
        ocr_strategy = OCR_STRATEGIES[strategy]
        extracted_text = ocr_strategy.extract_text_from_pdf(pdf_bytes)
        
        if ocr_cache:
            # Cache the result
            redis_client.set(pdf_hash, extracted_text)
        
        return {"text": extracted_text}

@app.get("/ocr/result/{task_id}")
async def ocr_status(task_id: str):
    """
    Endpoint to get the status of an OCR task using task_id.
    """
    task = AsyncResult(task_id, app=celery)

    if task.state == 'PENDING':
        return {"state": task.state, "status": "Task is pending..."}
    elif task.state == 'PROGRESS':
        return {"state": task.state, "status": "Task is in progress...", "progress": task.info.get('progress', 0)}
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

@app.post("/llama_test")
async def generate_llama(prompt: str):
    """
    Endpoint to generate text using Llama 3.1 model via the Ollama API.
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    response = requests.post(
        OLLAMA_API_URL,
        json={"model": "llama-3.1", "prompt": prompt}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to generate text with Ollama API")

    generated_text = response.json().get("generated_text", "")
    return {"generated_text": generated_text}