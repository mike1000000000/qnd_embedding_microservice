from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

class TextRequest(BaseModel):
    text: str

# Load the model
model = SentenceTransformer('model/all-MiniLM-L6-v2')

@app.post("/embeddings")
async def get_embeddings(request: TextRequest):
    embedding = model.encode(request.text)
    return {"embedding": embedding.tolist()}
