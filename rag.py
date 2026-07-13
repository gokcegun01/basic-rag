import os
import shutil
import json as json_parser
import chromadb
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter

load_dotenv()
client = genai.Client(api_key=os.getenv("API_KEY"))
get_emb = lambda t: client.models.embed_content(model="gemini-embedding-2", contents=t).embeddings[0].values

app = FastAPI(title="Gemini RAG API Service")
doc_converter = DocumentConverter()

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="rag_documents")

# --- Pydantic Data Models ---
class QueryModel(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, description="The query string to search in the documents")

class StructuredRAGResponse(BaseModel):
    answer: str = Field(..., description="The main answer grounded strictly in the provided document context")
    summary_sentence: str = Field(..., description="A single-sentence concise summary of the main answer")
    confidence_score: str = Field(..., description="The confidence level of finding the answer within the docs: High, Medium, or Low")

class FinalAPIResponse(StructuredRAGResponse):
    source: str = Field(..., description="The filename of the document where the answer was found")


# --- Core RAG Logic ---
def sync_chroma_db():
    if collection.count() > 0:
        return True
    if not os.path.exists("data"):
        return False
        
    documents = []
    embeddings = []
    ids = []
    metadatas = []
    id_counter = 0
    
    for file_name in os.listdir("data"):
        file_path = f"data/{file_name}"
        raw_chunks = []
        
        if file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_chunks = [p.strip() for p in f.read().split("\n\n") if p.strip()]
                
        elif file_name.endswith(".pdf"):
            conversion_result = doc_converter.convert(file_path)
            markdown_text = conversion_result.document.export_to_markdown()
            raw_chunks = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
            
        chunks = []
        for k in range(0, len(raw_chunks), 4):
            grouped_chunk = "\n\n".join(raw_chunks[k : k + 4])
            chunks.append(grouped_chunk)
            
        for chunk in chunks:
            documents.append(chunk)
            embeddings.append(get_emb(chunk)) 
            ids.append(f"doc_{id_counter}")
            metadatas.append({"file_name": file_name})
            id_counter += 1
            
    if documents:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    return True


# --- API Endpoints ---

# 1. Web Arayüzünü Servis Et (index.html dosyasını okur)
@app.get("/", response_class=HTMLResponse)
def read_root():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html file not found in root directory!")
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# 2. PDF/TXT Dosyası Yükleme Arayüzü
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    if not os.path.exists("data"):
        os.makedirs("data")
        
    file_path = f"data/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    global collection
    try:
        chroma_client.delete_collection(name="rag_documents")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name="rag_documents")
    
    sync_chroma_db()
    
    return {"message": f"'{file.filename}' processed and indexed successfully!"}

# 3. Soru Sorma Arayüzü (Geliştirilmiş & Kaynak Destekli)
@app.post("/ask", response_model=FinalAPIResponse)
def ask_question(data: QueryModel):
    if not sync_chroma_db():
        raise HTTPException(status_code=404, detail="'data' folder not found!")
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="No documents found in the database.")
        
    user_query = data.question
    query_vector = get_emb(user_query)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )
    
    retrieved_chunks = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    
    context = ""
    for chunk, meta in zip(retrieved_chunks, retrieved_metadatas):
        context += f"--- Document Source ({meta['file_name']}) ---\n{chunk}\n\n"
        
    prompt = f"""Answer the following question based ONLY on the provided context.
If the answer is not in the context, set the answer field to "I don't know based on the documents. 
Answer same language as question".

CONTEXT:
{context}

QUESTION:
{user_query}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuredRAGResponse,
            ),
        )
        
        source_file = retrieved_metadatas[0]['file_name'] if retrieved_metadatas else "Unknown Source"
        
        response_dict = json_parser.loads(response.text)
        response_dict["source"] = source_file
        
        return response_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))