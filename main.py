from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import extract_text_from_pdf, split_text, embed_chunks, build_faiss_index, retrieve_chunks
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess

app = FastAPI()

# Load model and PDF at startup
pdf_path = "sample-finance-statements.pdf"
text = extract_text_from_pdf(pdf_path)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for answers
cache = {}

# Store embeddings per chunking method
chunking_cache = {}

class QuestionRequest(BaseModel):
    question: str
    method: str = "words"  # Default method

def query_ollama(prompt, model="llama3"):
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, _ = process.communicate(prompt)
    return output

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    question = req.question.strip()
    method = req.method.strip().lower()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cache_key = f"{method}::{question}"
    if cache_key in cache:
        print("🔁 Using cached answer")
        return { "answer": cache[cache_key] }

    # Load or compute chunks/embeddings for this method
    if method not in chunking_cache:
        chunks = split_text(text, method=method)
        embeddings = np.array(embed_chunks(chunks, model))
        index = build_faiss_index(embeddings)
        chunking_cache[method] = { "chunks": chunks, "embeddings": embeddings, "index": index }
    else:
        chunks = chunking_cache[method]["chunks"]
        embeddings = chunking_cache[method]["embeddings"]
        index = chunking_cache[method]["index"]

    top_chunks = retrieve_chunks(question, chunks, embeddings, index, model, k=3)
    context = "\n\n".join(top_chunks)

    prompt = f"""
You are a helpful assistant. Use the information below to answer the user's question.

Context:
{context}

Question: {question}

Answer:
"""

    answer = query_ollama(prompt).strip()
    cache[cache_key] = answer

    return { "answer": answer,
            "method": method,
            "chunks_used": top_chunks
            }

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
