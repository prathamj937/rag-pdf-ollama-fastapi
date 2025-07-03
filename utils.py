import fitz
import re
import numpy as np
import faiss
import nltk

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, method='words', chunk_size=300, overlap=50):
    if method == 'lines':
        lines = text.split('\n')
        chunks = []
        for i in range(0, len(lines), chunk_size - overlap):
            chunk = "\n".join(lines[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    elif method == 'paragraphs':
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        for i in range(0, len(paragraphs), chunk_size - overlap):
            chunk = "\n\n".join(paragraphs[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    elif method == 'sentence':
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = " ".join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    else:  # default 'words'
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

def embed_chunks(chunks, model):
    return model.encode(chunks)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(question, chunks, embeddings, index, model, k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), k)
    return [chunks[i] for i in indices[0]]
