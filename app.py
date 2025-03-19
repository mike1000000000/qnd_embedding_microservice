from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import os
import secrets
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENV_FILE = ".env"

# Load env variables
if os.path.exists(ENV_FILE):
    load_dotenv()

def model_exists(model_path):
    """Check if the model directory exists and contains files"""
    return os.path.isdir(model_path) and bool(os.listdir(model_path))

def download_model(model_name, model_dir):
    """Download the model if it doesn't exist"""
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Downloading model: {model_name}...")
    model = SentenceTransformer(model_name, cache_folder=model_dir)
    logger.info(f"Model saved to: {model_dir}")
    return model

def update_env(key, value):
    """Append key=value to .env if not already set"""
    with open(ENV_FILE, "a+") as f:
        f.seek(0)  # Move to start to read existing content
        lines = f.readlines()
        if any(line.startswith(f"{key}=") for line in lines):
            return  # Key already exists, do nothing
        f.write(f"\n{key}={value}\n")    

# Load or generate API key
API_KEY = os.getenv("API_KEY", secrets.token_hex(32))
update_env("API_KEY", API_KEY)
logger.info(f"API_KEY: {API_KEY}")

# Load or set default model - relying on the env $MODEL variable, defaulting to 'all-MiniLM-L6-v2'
MODEL_NAME = os.getenv("MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_DIR = os.path.join(os.getcwd(), "model")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
logger.info(f"MODEL_NAME: {MODEL_NAME}")

# Check and load/download the model
if not model_exists(MODEL_PATH):
    model = download_model(MODEL_NAME, MODEL_PATH)
else:
    logger.info(f"Model already exists at {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)

class TextRequest(BaseModel):
    text: str

app = FastAPI()

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/embeddings")
async def get_embeddings(
    request: TextRequest,
    _ = Depends(verify_api_key)  # Protect this endpoint with API key authentication
):
    embedding = model.encode(request.text)
    return {"embedding": embedding.tolist()}
