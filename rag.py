import os
import shutil
import uuid
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

app = FastAPI(title="Enterprise Unified GenAI Copilot")
doc_converter = DocumentConverter()

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="rag_documents")

# =====================================================================================
# 🧠 [HAFIZA SİSTEMİ - BAŞLANGIÇ] CHROMADB HAFIZA ODASI TANIMLAMA
# =====================================================================================
ltm_collection = chroma_client.get_or_create_collection(name="user_long_term_memory")

class QueryModel(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    mode: str = Field("unified") 
    user_id: str = Field("user_101")
    history: list = Field(default=[])

class StructuredRAGResponse(BaseModel):
    answer: str
    summary_sentence: str
    confidence_score: str
    source: str

# =====================================================================================
# 🧠 [HAFIZA SİSTEMİ - ADIM 1] VERİTABANINA YENİ HAFIZA KAYDETME FONKSİYONU
# =====================================================================================
def save_to_long_term_memory(user_id: str, insight: str) -> bool:
    try:
        existing = ltm_collection.get(where={"user_id": user_id})
        if existing and insight in existing.get("documents", []):
            return False
            
        vector = get_emb(insight)
        ltm_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vector],
            documents=[insight],
            metadatas=[{"user_id": user_id}]
        )
        return True
    except Exception as e:
        print(f"LTM Storage Error: {e}")
        return False

# =====================================================================================
# 🧠 [HAFIZA SİSTEMİ - ADIM 2] VERİTABANINDAN GEÇMİŞ ANILARI SORGULAMA VE GETİRME
# =====================================================================================
def get_long_term_memory(user_id: str, query: str, limit: int = 3) -> str:
    try:
        if ltm_collection.count() == 0:
            return ""
        query_vector = get_emb(query)
        results = ltm_collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where={"user_id": user_id}
        )
        if results and results.get("documents") and results["documents"][0]:
            memories = "\n".join([f"- {doc}" for doc in results["documents"][0]])
            return f"\n[CRITICAL LONG-TERM MEMORY - CONFIRMED USER FACTS]:\n{memories}\n"
    except Exception:
        pass
    return ""

# =====================================================================================
# 🧠 [HAFIZA SİSTEMİ - ADIM 3] KULLANICI MESAJINDAN OTONOM BİLGİ AYIKLAMA MOTORU
# =====================================================================================
def auto_extract_and_save_memory(user_id: str, user_msg: str):
    extraction_prompt = f"""Analyze the following user message. If it contains a permanent personal fact, role, corporate affiliation, project detail, or preference about the user, extract it as a single, clear declarative sentence in the SAME language as the user's message.
If it contains no permanent information worth remembering long-term, reply ONLY with the word "NONE".

USER MESSAGE: "{user_msg}"
EXTRACTED FACT:"""
    try:
        res = client.models.generate_content(model="gemini-3.1-flash-lite", contents=extraction_prompt)
        extracted_text = res.text.strip()
        if extracted_text and "NONE" not in extracted_text.upper():
            save_to_long_term_memory(user_id=user_id, insight=extracted_text)
    except Exception as e:
        print(f"Autonomous memory extraction failed: {e}")

def serialize_history(conv_list) -> list:
    serialized = []
    for item in conv_list:
        try:
            if hasattr(item, "model_dump"):
                d = item.model_dump(exclude_none=True)
            elif isinstance(item, dict):
                d = item
            else:
                d = str(item)
            pure_json = json_parser.loads(json_parser.dumps(d, default=str))
            serialized.append(pure_json)
        except Exception:
            pass
    return serialized

def get_coupon_discount(coupon_code: str) -> dict:
    """Sistemdeki aktif kupon kodlarının indirim oranlarını döner."""
    coupons = {"yaz50": 0.50, "okan20": 0.20, "staj100": 1.00}
    code_lower = coupon_code.lower().strip()
    discount = coupons.get(code_lower, 0.0)
    return {"coupon": coupon_code, "discount_rate": f"%{int(discount * 100)}", "valid": discount > 0}

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
    """Sistem veritabanına sorgu atarak aktif kullanıcıya ait profil detaylarını getirir."""
    live_database = {
        "user_101": {"name": "Gökçe Naz Gün", "role": "Software Engineering Intern", "department": "Generative AI", "status": "Active"},
        "user_102": {"name": "Hakan Bey", "role": "Mentor / Senior Software Architect", "department": "Generative AI", "status": "Active"}
    }
    normalized_input = user_id.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    for key, data in live_database.items():
        if normalized_input == key.replace("_", ""):
            return {"found": True, "user_data": data}
    return {"found": False, "error": f"ID '{user_id}' not found in personnel registry."}

def sync_chroma_db():
    if collection.count() > 0:
        return True
    if not os.path.exists("data"):
        return False
        
    documents, embeddings, ids, metadatas = [], [], [], []
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
            
        chunks = ["\n\n".join(raw_chunks[k : k + 4]) for k in range(0, len(raw_chunks), 4)]
            
        for chunk in chunks:
            documents.append(chunk)
            embeddings.append(get_emb(chunk)) 
            ids.append(f"doc_{id_counter}")
            metadatas.append({"file_name": file_name})
            id_counter += 1
            
    if documents:
        collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return True

def search_web_free(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "".join([f"Source Link: {r.get('href', 'N/A')}\nTitle: {r.get('title', '')}\nSnippet: {r.get('body', '')}\n\n" for r in results])
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
    return ""

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

@app.post("/ask")
def ask_question(data: QueryModel):
    try:
        user_query = data.question

        # =====================================================================================
        # 🧠 [HAFIZA SİSTEMİ - CANLI TETİKLEME] HER SORUDA ARKA PLAN HAFIZA AKIŞININ ÇALIŞMASI
        # =====================================================================================
        auto_extract_and_save_memory(user_id=data.user_id, user_msg=user_query)
        ltm_context = get_long_term_memory(user_id=data.user_id, query=user_query)
        
        # =====================================================================================
        # 🆔 YAZIM TARZI SERBESTLİK KURALI (SPACING & PUNCTUATION NEUTRALITY)
        # =====================================================================================
        user_id_rule = """
        CRITICAL USER ID RESOLUTION RULE:
        - User IDs may be formatted with different spaces or underscores (e.g., 'user 101', 'user101', 'user_101').
        - You MUST treat 'user 101', 'user101', and 'user_101' as the EXACT SAME entity and ID. Spacing and writing style do not matter.
        """

        # =====================================================================================
        # ⏳ [MİMARİ DÜZELTME] KISA VADELİ HAFIZAYI (HISTORY) İNŞA ETME VE ASLA KAYBETMEME
        # =====================================================================================
        conversation = []
        if data.history:
            for msg in data.history:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    parts_list = msg.get("parts", [])
                    sdk_parts = []
                    for p in parts_list:
                        if isinstance(p, dict):
                            if "text" in p and p["text"]:
                                sdk_parts.append(types.Part.from_text(text=p["text"]))
                            elif "function_call" in p and p["function_call"]:
                                fc = p["function_call"]
                                sdk_parts.append(types.Part(function_call=types.FunctionCall(name=fc.get("name"), args=fc.get("args"))))
                            elif "function_response" in p and p["function_response"]:
                                fr = p["function_response"]
                                sdk_parts.append(types.Part(function_response=types.FunctionResponse(name=fr.get("name"), response=fr.get("response"))))
                    if sdk_parts:
                        conversation.append(types.Content(role=role, parts=sdk_parts))

        # O anki güncel soruyu hafıza listesinin sonuna ekle
        conversation.append(types.Content(role="user", parts=[types.Part.from_text(text=user_query)]))
        
        # 🎯 TEMEL KORUYUCU PROMPT TALİMATLARI
        system_inst = f"""You are a professional Enterprise AI Copilot. Answer in the same language as the user's question.
        {user_id_rule}
        
        STRICT GROUNDING RULES:
        - You must ONLY use facts directly and explicitly mentioned in the provided Document Context or Long-Term Memory.
        - NEVER assume, extrapolate, or invent any details (such as names, scores, levels, dates, or project status) if they are not explicitly written.
        - If the user is asking about a document, certificate, or CV and the information is missing or unreadable, state clearly that it is not found. Do NOT use your parametric general knowledge to fill in blanks or guess realistic values (like standard test scores).
        """
        
        if ltm_context:
            system_inst += f"\n[VALIDATED LONG-TERM MEMORY]:\n{ltm_context}"

        # =====================================================================================
        # ⚙️ 1. AŞAMA: ARAÇ (TOOL) TETİKLEME VE ÇALIŞTIRMA MANTIĞI
        # =====================================================================================
        try:
            all_tools = [get_coupon_discount, get_all_active_coupons, list_indexed_pdfs, query_user_profile]
            config = types.GenerateContentConfig(
                system_instruction=system_inst,
                tools=all_tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
            response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=conversation, config=config)
            
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
                tool_response_part = types.Part.from_function_response(name=func_name, response=tool_result)
                conversation.append(types.Content(role="tool", parts=[tool_response_part]))
                
                final_response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=StructuredRAGResponse
                    )
                )
                
                conversation.append(final_response.candidates[0].content)
                try:
                    response_dict = json_parser.loads(final_response.text)
                except Exception:
                    response_dict = {
                        "answer": final_response.text,
                        "summary_sentence": "Sistem aracı başarıyla çalıştırıldı.",
                        "confidence_score": "High"
                    }
                
                return {
                    "answer": response_dict.get("answer", "İşlem başarıyla tamamlandı."),
                    "summary_sentence": response_dict.get("summary_sentence", "Araç tetiklendi."),
                    "confidence_score": response_dict.get("confidence_score", "High"),
                    "source": response_dict.get("source") or f"Active System Tool ({func_name})",
                    "history": serialize_history(conversation)
                }

        except Exception as e:
            print(f"Tool routing check bypassed or failed: {e}")
            pass

        # =====================================================================================
        # 📂 2. AŞAMA: DÖKÜMAN ANALİZİ VE YEREL RAG Component'i
        # =====================================================================================
        context = ""
        source_file = "Local Document"
        
        local_docs_available = False
        try:
            if sync_chroma_db() and collection.count() > 0:
                local_docs_available = True
        except Exception:
            pass
            
        if local_docs_available:
            try:
                query_vector = get_emb(user_query)
                # 📍 n_results değerini 6'ya çekerek döküman çakışmalarını egale ediyoruz
                results = collection.query(query_embeddings=[query_vector], n_results=6)
                retrieved_chunks = results.get("documents", [[]])[0]
                retrieved_metadatas = results.get("metadatas", [[]])[0]
                
                if retrieved_chunks:
                    context = "".join([f"--- Document Source ({meta['file_name']}) ---\n{chunk}\n\n" for chunk, meta in zip(retrieved_chunks, retrieved_metadatas)])
                    source_file = retrieved_metadatas[0]['file_name'] if retrieved_metadatas else "Local Document"
                    
                    print("\n--- [DEBUG] RETRIEVED CONTEXT START ---")
                    print(context)
                    print("--- [DEBUG] RETRIEVED CONTEXT END ---\n")
            except Exception as e:
                print(f"Local RAG components search failed: {e}")
                pass

        # 🎯 KULLANICI DOĞRUDAN BELGE Mİ HEDEFLİYOR KONTROLÜ
        is_document_targeted = any(k in user_query.lower() for k in ["pdf", "belge", "dosya", "cv", "sertifika", "itep"])
        
        # Eğer döküman verisi bulunduysa VEYA kullanıcı doğrudan bir dökümanı sorguluyorsa akışı kilitle:
        if context or is_document_targeted:
            # RAG döküman içeriğini ana system instruction kümesine enjekte ediyoruz
            system_inst += f"\n\n[DOCUMENT CONTEXT]:\n{context if context else 'No explicitly matching document sections found in the database.'}"
            
            if is_document_targeted:
                system_inst += "\nCRITICAL: User is specifically asking about a file/document. Do NOT use Web Search or Parametric Knowledge fallbacks if information is missing. Strictly stick to the Document Context."

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=conversation,  # 📍 ARTIK GEÇMİŞİ UNUTMUYOR, TÜM LİSTEYİ GÖNDERİYORUZ!
                config=types.GenerateContentConfig(
                    system_instruction=system_inst,
                    response_mime_type="application/json",
                    response_schema=StructuredRAGResponse
                ),
            )
            
            try:
                response_dict = json_parser.loads(response.text)
            except Exception:
                response_dict = {
                    "answer": response.text,
                    "summary_sentence": "Belge analizi sağlandı.",
                    "confidence_score": "Medium",
                    "source": source_file
                }
            
            confidence = response_dict.get("confidence_score", "").upper()
            answer_text = response_dict.get("answer", "").lower()
            is_unknown = any(x in answer_text for x in ["don't know", "bilmiyorum", "not found", "i do not know", "ulaşılamadı"])
            
            # 📍 MUTLAK DURDURMA KAPISI: Eğer kullanıcı belge hedeflediyse sonuç Low bile olsa internete akıtma, cevabı ver!
            if not is_unknown and confidence != "LOW" or is_document_targeted:
                conversation.append({"role": "model", "parts": [{"text": response_dict.get("answer", "")}]})
                return {
                    "answer": response_dict.get("answer", ""),
                    "summary_sentence": response_dict.get("summary_sentence", ""),
                    "confidence_score": response_dict.get("confidence_score", "High"),
                    "source": response_dict.get("source") or source_file,
                    "history": serialize_history(conversation)
                }

        # =====================================================================================
        # 🧠 3. AŞAMA: MODEL PARAMETRİK SEÇİM ALANI (Yalnızca jenerik genel kültür soruları için)
        # =====================================================================================
        try:
            internal_prompt_inst = system_inst + """\nAnswer using your own general knowledge or Long-Term Memory.
            DIRECTIONS FOR SOURCE FIELD:
            - If you answer using the user facts inside Long-Term Memory, you MUST set the "source" field to "Long-Term Memory".
            - Otherwise, set it to "Gemini Knowledge (Parametric)"."""
            
            internal_response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=conversation,
                config=types.GenerateContentConfig(
                    system_instruction=internal_prompt_inst,
                    response_mime_type="application/json",
                    response_schema=StructuredRAGResponse
                ),
            )
            
            try:
                internal_dict = json_parser.loads(internal_response.text)
            except Exception:
                internal_dict = {
                    "answer": internal_response.text,
                    "summary_sentence": "Genel bilgi sistemi.",
                    "confidence_score": "High",
                    "source": "Gemini Knowledge (Parametric)"
                }
                
            internal_confidence = internal_dict.get("confidence_score", "").upper()
            internal_answer = internal_dict.get("answer", "").lower()
            internal_unknown = any(x in internal_answer for x in ["don't know", "bilmiyorum", "not found", "i do not know"])
            
            if not internal_unknown and internal_confidence in ["HIGH", "MEDIUM"]:
                conversation.append({"role": "model", "parts": [{"text": internal_dict.get("answer", "")}]})
                return {
                    "answer": internal_dict.get("answer", ""),
                    "summary_sentence": internal_dict.get("summary_sentence", ""),
                    "confidence_score": internal_dict.get("confidence_score", "High"),
                    "source": internal_dict.get("source") or "Gemini Knowledge (Parametric)",
                    "history": serialize_history(conversation)
                }
                
        except Exception as e:
            print(f"Internal knowledge fallback failed: {e}")
            pass

        # =====================================================================================
        # 🌐 4. AŞAMA: İNTERNET ARAMA MOTORU (WEB FALLBACK)
        # =====================================================================================
        try:
            web_context = search_web_free(user_query)
            if not web_context:
                return {
                    "answer": "Aradığınız bilgi ne yerel belgelerde, ne yapay zeka hafızasında ne de internette bulunabildi.",
                    "summary_sentence": "Bilgi hiçbir kaynakta bulunamadı.",
                    "confidence_score": "Low",
                    "source": "Not Found",
                    "history": serialize_history(conversation)
                }

            web_prompt_inst = system_inst + f"""\nAnswer strictly based on the provided Web Search Context or Long-Term Memory.
            
            [WEB SEARCH CONTEXT]:
            {web_context}
            
            DIRECTIONS FOR SOURCE FIELD:
            - If you answer using the user facts inside Long-Term Memory, you MUST set the "source" field to "Long-Term Memory".
            - Otherwise, set it to "DuckDuckGo Search (Web)"."""
            
            structured_response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=conversation,
                config=types.GenerateContentConfig(
                    system_instruction=web_prompt_inst,
                    response_mime_type="application/json",
                    response_schema=StructuredRAGResponse
                ),
            )
            
            try:
                web_response_dict = json_parser.loads(structured_response.text)
            except Exception:
                web_response_dict = {
                    "answer": structured_response.text,
                    "summary_sentence": "İnternet arama havuzu.",
                    "confidence_score": "High",
                    "source": "DuckDuckGo Search (Web)"
                }
                
            conversation.append({"role": "model", "parts": [{"text": web_response_dict.get("answer", "")}]})
            return {
                "answer": web_response_dict.get("answer", ""),
                "summary_sentence": web_response_dict.get("summary_sentence", ""),
                "confidence_score": web_response_dict.get("confidence_score", "High"),
                "source": web_response_dict.get("source") or "DuckDuckGo Search (Web)",
                "history": serialize_history(conversation)
            }
            
        except Exception as e:
            raise Exception(f"Fallback chain edge failure: {str(e)}")

    except Exception as general_error:
        return {
            "answer": f"İşlem tamamlanamadı: {str(general_error)}",
            "summary_sentence": "Sistem genelinde beklenmeyen bir operasyon hatası yakalandı.",
            "confidence_score": "Low",
            "source": "System Error Safetynet",
            "history": []
        }