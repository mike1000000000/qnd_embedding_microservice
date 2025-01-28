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
    API_KEY = secrets.token_hex(32)  # Generate a 32-byte secure key
    with open(".env", "a") as f:  # Append the new key to the .env file
        f.write(f"\nAPI_KEY={API_KEY}\n")
        print("Generated Key")

print(f"API_KEY: {API_KEY}")



class TextRequest(BaseModel):
    text: str

app = FastAPI()

# Load the model
model = SentenceTransformer('model/all-MiniLM-L6-v2')

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
