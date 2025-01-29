from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import os
import secrets
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Load or generate the API key
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    try:
        API_KEY = secrets.token_hex(32) 

        # Append new key to .env file
        with open(".env", "a") as f:  
            f.write(f"\nAPI_KEY={API_KEY}\n")
        print("Generated Key")
    except IOError:
        print("Warning: Unable to write to .env. Set API_KEY manually.")

print(f"API_KEY: {API_KEY}")

class TextRequest(BaseModel):
    text: str

app = FastAPI()

# Load the model
model_path = os.path.abspath(os.path.join(os.getcwd(), "model/all-MiniLM-L6-v2"))
model = SentenceTransformer(model_path)

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
