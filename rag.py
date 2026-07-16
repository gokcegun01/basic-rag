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
from duckduckgo_search import DDGS

load_dotenv()
client = genai.Client(api_key=os.getenv("API_KEY"))
get_emb = lambda t: client.models.embed_content(model="gemini-embedding-2", contents=t).embeddings[0].values

app = FastAPI(title="Gemini RAG API Service")
doc_converter = DocumentConverter()

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="rag_documents")

# --- Pydantic Data Models ---
class QueryModel(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, description="The query string to search")
    mode: str = Field("rag", description="Query mode: 'rag' or 'coupon'") 

class StructuredRAGResponse(BaseModel):
    answer: str = Field(..., description="The main answer based on the selected mode context")
    summary_sentence: str = Field(..., description="A single-sentence concise summary of the main answer")
    confidence_score: str = Field(..., description="Confidence level: High, Medium, or Low")

class FinalAPIResponse(StructuredRAGResponse):
    source: str = Field(..., description="The source of the answer (file name, web search, or system tool)")


# --- 🔌 Active LLM System Tools (Function Calling) ---

def get_coupon_discount(coupon_code: str) -> dict:
    """Sistemdeki aktif kupon kodlarının indirim oranlarını döner.
    
    Args:
        coupon_code: Sorgulanacak olan indirim kuponu kodu (örn: YAZ50, OKAN20, STAJ100)
    """
    coupons = {
        "yaz50": 0.50,    # %50 indirim
        "okan20": 0.20,   # %20 indirim
        "staj100": 1.00   # %100 indirim
    }
    
    code_lower = coupon_code.lower().strip()
    discount = coupons.get(code_lower, 0.0)
    return {
        "coupon": coupon_code, 
        "discount_rate": f"%{int(discount * 100)}", 
        "valid": discount > 0
    }

def get_all_active_coupons() -> dict:
    """Sistemde tanımlı olan tüm aktif indirim kuponu kodlarını liste halinde döner."""
    return {"active_coupons": ["YAZ50", "OKAN20", "STAJ100"]}

def list_indexed_pdfs() -> dict:
    """Sistemde yüklü olan ve yapay zekanın erişebildiği güncel PDF/TXT dosyalarının isimlerini liste halinde döner."""
    if not os.path.exists("data"):
        return {"files": []}
    files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f)) and not f.startswith(".")]
    return {"files": files}

def query_user_profile(user_id: str) -> dict:
    """Sistem veritabanına güvenli sorgu atarak, verilen kullanıcı ID'sine ait profil, rol ve departman bilgilerini çeker.
    
    Args:
        user_id: Sorgulanacak kullanıcının benzersiz kimlik kodu (örn: user_101, user_102)
    """
    mock_database = {
        "user_101": {
            "name": "Gökçe Gün",
            "role": "Software Engineering Intern",
            "department": "Generative AI",
            "status": "Active"
        },
        "user_102": {
            "name": "Hakan Bey",
            "role": "Mentor / Senior Software Architect",
            "department": "Generative AI",
            "status": "Active"
        }
    }
    
    # Gelişmiş Eşleştirme: Kullanıcı 'user101' yazsa bile 'user_101' ile eşleştirir
    normalized_input = user_id.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    
    for key, data in mock_database.items():
        normalized_key = key.replace("_", "")
        if normalized_input == normalized_key:
            return {"found": True, "user_data": data}
            
    return {"found": False, "error": f"ID '{user_id}' veritabanında bulunamadı."}


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


# --- Free Web Search Helper ---
def search_web_free(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                context = ""
                for r in results:
                    context += f"Source Link: {r.get('href', 'N/A')}\nTitle: {r.get('title', '')}\nSnippet: {r.get('body', '')}\n\n"
                return context
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
    return ""


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
def read_root():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html file not found!")
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/files")
def list_files():
    if not os.path.exists("data"):
        return []
    return [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f)) and not f.startswith(".")]

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

@app.post("/ask", response_model=FinalAPIResponse)
def ask_question(data: QueryModel):
    user_query = data.question
    
    # ==========================================
    # MOD 1: AJAN VE SİSTEM ARAÇLARI MODU (Agentic Tool Calling)
    # ==========================================
    if data.mode == "coupon":
        try:
            all_tools = [get_coupon_discount, get_all_active_coupons, list_indexed_pdfs, query_user_profile]
            
            conversation = [types.Content(role="user", parts=[types.Part.from_text(text=user_query)])]
            config = types.GenerateContentConfig(
                tools=all_tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=conversation,
                config=config
            )
            
            if response.function_calls:
                func_call = response.function_calls[0]
                func_name = func_call.name
                func_args = func_call.args or {}
                
                if func_name == "get_coupon_discount":
                    tool_result = get_coupon_discount(coupon_code=func_args.get("coupon_code"))
                elif func_name == "get_all_active_coupons":
                    tool_result = get_all_active_coupons()
                elif func_name == "list_indexed_pdfs":
                    tool_result = list_indexed_pdfs()
                elif func_name == "query_user_profile":
                    tool_result = query_user_profile(user_id=func_args.get("user_id"))
                else:
                    tool_result = {"error": f"Unknown function: {func_name}"}
                
                conversation.append(response.candidates[0].content)
                
                tool_response_part = types.Part.from_function_response(
                    name=func_name,
                    response=tool_result
                )
                conversation.append(types.Content(role="tool", parts=[tool_response_part]))
                
                final_response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=StructuredRAGResponse
                    )
                )
                
                response_dict = json_parser.loads(final_response.text)
                response_dict["source"] = f"Active System Tool ({func_name})"
                return response_dict
                
            else:
                structuring_prompt = f"Structure this text into the required JSON schema:\n\n{response.text}"
                fallback_resp = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=structuring_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=StructuredRAGResponse
                    )
                )
                response_dict = json_parser.loads(fallback_resp.text)
                
                direct_answer = response_dict.get("answer", "").lower()
                direct_is_unknown = any(x in direct_answer for x in ["don't know", "dont know", "bilmiyorum", "bulunamadı", "not found", "no information", "bilgi yok", "i do not know"])
                
                if direct_is_unknown:
                    response_dict["source"] = "Not Found"
                else:
                    response_dict["source"] = "System Agent (Direct)"
                    
                return response_dict
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent workflow failed: {str(e)}")

    # ==========================================
    # MOD 2: PDF & WEB ARAMA (RAG Pipeline with Parametric Fallback)
    # ==========================================
    else:
        local_docs_available = False
        try:
            if sync_chroma_db() and collection.count() > 0:
                local_docs_available = True
        except Exception:
            pass
            
        # Adım 1: PDF Belgelerinde (Local RAG) ara
        if local_docs_available:
            query_vector = get_emb(user_query)
            results = collection.query(query_embeddings=[query_vector], n_results=3)
            
            retrieved_chunks = results.get("documents", [[]])[0]
            retrieved_metadatas = results.get("metadatas", [[]])[0]
            
            if retrieved_chunks:
                context = ""
                for chunk, meta in zip(retrieved_chunks, retrieved_metadatas):
                    context += f"--- Document Source ({meta['file_name']}) ---\n{chunk}\n\n"
                    
                prompt = f"""Answer the following question based ONLY on the provided context.
If the answer is NOT strictly contained in the context, you MUST set the answer field to "I don't know based on the documents" and set confidence_score to "Low".
Answer in the same language as the question.

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
                    
                    response_dict = json_parser.loads(response.text)
                    confidence = response_dict.get("confidence_score", "").upper()
                    answer_text = response_dict.get("answer", "").lower()
                    is_unknown = any(x in answer_text for x in ["don't know", "dont know", "bilmiyorum", "bulunamadı", "not found", "no information", "bilgi yok", "i do not know"])
                    
                    if not is_unknown and confidence != "LOW":
                        source_file = retrieved_metadatas[0]['file_name'] if retrieved_metadatas else "Unknown Source"
                        response_dict["source"] = source_file
                        return response_dict
                        
                except Exception as e:
                    print(f"Local query failed: {e}")
                    pass

        # Adım 2: Gemini'nin Kendi Genel Bilgisi (Parametric Memory) ile Yanıtlamayı Dene
        # Böylece internete çıkmadan önce genel kültür sorularını kendi hafızasından çözer.
        try:
            internal_prompt = f"""Answer the following question using your own general knowledge.
Answer in the same language as the question.
If you know the answer clearly, provide it and set confidence_score to "High" or "Medium".
If you do not know the answer or if it requires real-time/current information that you cannot be sure of, state that you don't know and set confidence_score to "Low".

QUESTION:
{user_query}
"""
            internal_response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=internal_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=StructuredRAGResponse,
                ),
            )
            
            internal_dict = json_parser.loads(internal_response.text)
            internal_confidence = internal_dict.get("confidence_score", "").upper()
            internal_answer = internal_dict.get("answer", "").lower()
            
            internal_unknown = any(x in internal_answer for x in ["don't know", "dont know", "bilmiyorum", "bulunamadı", "not found", "no information", "bilgi yok", "i do not know"])
            
            if not internal_unknown and internal_confidence in ["HIGH", "MEDIUM"]:
                internal_dict["source"] = "Gemini Knowledge (Parametric)"
                return internal_dict
                
        except Exception as e:
            print(f"Internal general knowledge check failed: {e}")
            pass

        # Adım 3: Son Çare Olarak Canlı Web Araması (DuckDuckGo Fallback)
        try:
            web_context = search_web_free(user_query)
            if not web_context:
                return {
                    "answer": "Aradığınız bilgi ne yerel belgelerde, ne yapay zeka hafızasında ne de internette bulunabildi.",
                    "summary_sentence": "Bilgi hiçbir kaynakta bulunamadı.",
                    "confidence_score": "Low",
                    "source": "Not Found"
                }

            structuring_prompt = f"""You are a highly capable AI assistant. Answer the user's question based strictly on the provided Web Search Context.
Answer in the same language as the question.
If the information is not present in the context, set confidence_score to "Low" and state that you don't know.

WEB SEARCH CONTEXT:
{web_context}

QUESTION:
{user_query}
"""
            structured_response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=structuring_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=StructuredRAGResponse,
                ),
            )
            
            web_response_dict = json_parser.loads(structured_response.text)
            
            web_answer_text = web_response_dict.get("answer", "").lower()
            web_is_unknown = any(x in web_answer_text for x in ["don't know", "dont know", "bilmiyorum", "bulunamadı", "not found", "no information", "bilgi yok", "i do not know"])
            
            if web_is_unknown:
                web_response_dict["source"] = "Not Found"
            else:
                web_response_dict["source"] = "DuckDuckGo Search (Web)"
                
            return web_response_dict
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Web fallback failed: {str(e)}")