import os
import shutil
import uuid
import re
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

# Google GenAI Asenkron Client
client = genai.Client(api_key=os.getenv("API_KEY"))

# Synchronous helper for embeddings
def get_emb(text: str):
    return client.models.embed_content(model="gemini-embedding-2", contents=text).embeddings[0].values

app = FastAPI(title="Enterprise Unified GenAI Copilot (Optimized)")
doc_converter = DocumentConverter()

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="rag_documents")
ltm_collection = chroma_client.get_or_create_collection(name="user_long_term_memory")

# =====================================================================================
# 📋 [SCHEMAS & MODELS]
# =====================================================================================
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

class IntentAnalysis(BaseModel):
    intent: str = Field(..., description="Target routing pipeline. Must be one of: 'tool', 'rag', 'parametric', 'web'")
    target_tool: str = Field(None, description="The specific tool name to invoke if intent is 'tool'. Options: 'get_coupon_discount', 'get_all_active_coupons', 'list_indexed_pdfs', 'query_user_profile'")
    tool_argument: str = Field(None, description="The raw string value/argument needed for the target tool (e.g., the coupon code or the user id) if applicable.")
    is_document_targeted: bool = Field(..., description="True if the user query specifically addresses an uploaded file, document, PDF, CV, or iTEP certificate.")
    refined_query: str = Field(..., description="An optimized search query stripped of typo irregularities, conversational padding, and user ID spaces.")

# =====================================================================================
# 🛠️ [SYSTEM TOOLS & WORKSPACE INTEGRATIONS]
# =====================================================================================
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
    coupons = {"yaz50": 0.50, "okan20": 0.20, "staj100": 1.00}
    code_lower = coupon_code.lower().strip()
    discount = coupons.get(code_lower, 0.0)
    return {"coupon": coupon_code, "discount_rate": f"%{int(discount * 100)}", "valid": discount > 0}

def get_all_active_coupons() -> dict:
    return {"active_coupons": ["YAZ50", "OKAN20", "STAJ100"]}

def list_indexed_pdfs() -> dict:
    if not os.path.exists("data"):
        return {"files": []}
    files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f)) and not f.startswith(".")]
    return {"files": files}

def query_user_profile(user_id: str) -> dict:
    live_database = {
        "user_101": {"name": "Gökçe Naz Gün", "role": "Software Engineering Intern", "department": "Generative AI", "status": "Active"},
        "user_102": {"name": "Hakan Bey", "role": "Mentor / Senior Software Architect", "department": "Generative AI", "status": "Active"}
    }
    normalized_input = user_id.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    for key, data in live_database.items():
        if normalized_input == key.replace("_", ""):
            return {"found": True, "user_data": data}
    return {"found": False, "error": f"ID '{user_id}' not found in personnel registry."}

def search_web_free(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "".join([f"Source Link: {r.get('href', 'N/A')}\nTitle: {r.get('title', '')}\nSnippet: {r.get('body', '')}\n\n" for r in results])
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
    return ""

def index_single_file(file_path: str, file_name: str):
    """Artımlı (incremental) dosya chunking ve indeksleme helper'ı."""
    raw_chunks = []
    if file_name.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_chunks = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    elif file_name.endswith(".pdf"):
        conversion_result = doc_converter.convert(file_path)
        markdown_text = conversion_result.document.export_to_markdown()
        raw_chunks = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]

    chunks = ["\n\n".join(raw_chunks[k : k + 4]) for k in range(0, len(raw_chunks), 4)]
    if chunks:
        documents, embeddings, ids, metadatas = [], [], [], []
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            embeddings.append(get_emb(chunk))
            ids.append(f"{file_name}_{idx}_{uuid.uuid4().hex[:6]}")
            metadatas.append({"file_name": file_name})
        collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

# =====================================================================================
# 🛡️ [GUARDRAILS & MEMORY HELPERS]
# =====================================================================================
def run_input_guardrails(user_query: str) -> dict:
    if re.search(r'\b\d{11,16}\b', user_query):
        return {
            "safe": False, 
            "reason": "Your request was blocked for security reasons due to potential sensitive personally identifiable information (PII) leakage."
        }
        
    injection_triggers = [
        "ignore previous instructions", "system prompt", "forget the rules above",
        "your new instructions are", "disable all instructions", "override all instructions"
    ]
    if any(trigger in user_query.lower() for trigger in injection_triggers):
        return {
            "safe": False, 
            "reason": "Request blocked by security guardrails due to detected system manipulation (Prompt Injection)."
        }
    return {"safe": True, "reason": "Clear"}

async def save_to_long_term_memory(user_id: str, insight: str) -> bool:
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

async def get_long_term_memory(user_id: str, query: str, limit: int = 3) -> str:
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

async def auto_extract_and_save_memory(user_id: str, user_msg: str):
    extraction_prompt = f"""Analyze the following user message. If it contains a permanent personal fact, role, corporate affiliation, project detail, or preference about the user, extract it as a single, clear declarative sentence in the SAME language as the user's message.
If it contains no permanent information worth remembering long-term, reply ONLY with the word "NONE".

USER MESSAGE: "{user_msg}"
EXTRACTED FACT:"""
    try:
        res = await client.aio.models.generate_content(model="gemini-3.1-flash-lite", contents=extraction_prompt)
        extracted_text = res.text.strip()
        if extracted_text and "NONE" not in extracted_text.upper():
            await save_to_long_term_memory(user_id=user_id, insight=extracted_text)
    except Exception as e:
        print(f"Autonomous memory extraction failed: {e}")

# =====================================================================================
# 🤖 [4-AGENT SEQUENTIAL PIPELINE - ASYNC EXECUTION]
# =====================================================================================
async def agent_1_intent_router(user_query: str, history_summary: str, id_rule: str) -> IntentAnalysis:
    router_prompt = f"""You are 'Agent 1: Intent Router'. Your sole job is to parse the user's incoming query and the short-term conversation context to pick the exact execution pipeline.

    {id_rule}

    AVAILABLE INTENT CLASSIFICATIONS:
    - 'tool': Use this if the query asks to fetch coupons, check specific coupon discounts, list active document indexes, or look up live user profiles (e.g., user_101, user 101, user101, user_102).
    - 'rag': Use this if the user asks about document metrics, specific scores, certificates, CV details, or contents expected inside uploaded files.
    - 'parametric': General queries, greetings, core logic discussions, code configuration syntax, or assistance.
    - 'web': Public real-world dynamic facts, general internet lookups outside local documents.

    DETAILED SYSTEM TOOLS LIST:
    - get_coupon_discount (Requires a coupon code string)
    - get_all_active_coupons (No arguments needed)
    - list_indexed_pdfs (No arguments needed)
    - query_user_profile (Requires a target user ID string like 'user_101')

    CRITICAL RULES:
    1. If words like 'pdf', 'file', 'document', 'cv', 'itep', or 'certificate' are found, classify 'is_document_targeted' as True.
    2. Clean the query into 'refined_query' to bypass typo or syntax constraints.

    CONVERSATION CONTEXT DETECTED:
    {history_summary}

    CURRENT USER QUERY:
    "{user_query}"

    Generate the parsing routing plan according to the structural schema.
    """
    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=router_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IntentAnalysis
        )
    )
    return IntentAnalysis.model_validate_json(response.text)

async def agent_2_data_executor(routing: IntentAnalysis, conversation_context: list, ltm: str) -> dict:
    context_data = ""
    resolved_source = "Factual Knowledge Draft"
    
    if routing.intent == "tool":
        t_name = routing.target_tool
        t_arg = routing.tool_argument or ""
        
        if t_name == "get_coupon_discount":
            context_data = str(get_coupon_discount(coupon_code=t_arg))
            resolved_source = "System Tool (get_coupon_discount)"
        elif t_name == "get_all_active_coupons":
            context_data = str(get_all_active_coupons())
            resolved_source = "System Tool (get_all_active_coupons)"
        elif t_name == "list_indexed_pdfs":
            context_data = str(list_indexed_pdfs())
            resolved_source = "System Tool (list_indexed_pdfs)"
        elif t_name == "query_user_profile":
            context_data = str(query_user_profile(user_id=t_arg))
            resolved_source = "System Tool (query_user_profile)"
        else:
            context_data = "{'error': 'Target system tool match failed.'}"
            
    elif routing.intent == "rag" or routing.is_document_targeted:
        try:
            if collection.count() > 0:
                query_vector = get_emb(routing.refined_query)
                results = collection.query(query_embeddings=[query_vector], n_results=6)
                chunks = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                
                if chunks:
                    context_data = "".join([f"--- Content Chunk ({m['file_name']}) ---\n{c}\n\n" for c, m in zip(chunks, metadatas)])
                    resolved_source = metadatas[0]['file_name']
                else:
                    context_data = "No matching text entries found in the vector storage pool."
                    resolved_source = "Not Found"
        except Exception as ex:
            context_data = f"Failed to successfully fetch vector context nodes: {str(ex)}"
            
    elif routing.intent == "web":
        context_data = search_web_free(routing.refined_query)
        resolved_source = "DuckDuckGo Search (Web)"

    draft_prompt = f"""You are 'Agent 2: Data Executor & RAG Specialist'. Your job is to compile a raw text draft solution answering the query based on the retrieved environment context blocks. 
    Answer in the same language as the user query.

    LONG-TERM MEMORY DATA:
    {ltm}

    RAW DATA CONTEXT CONSOLE:
    {context_data}

    Is this answer bound to a document verification restriction?: {routing.is_document_targeted}
    Refined Core Query: {routing.refined_query}

    Write a detailed, factually accurate raw response text layout. Do not write JSON structure yet, provide only raw response content.
    DRAFT TEXT OUTCOME:"""

    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=conversation_context + [types.Content(role="user", parts=[types.Part.from_text(text=draft_prompt)])]
    )
    
    return {"draft": response.text, "context": context_data, "source": resolved_source}

async def agent_3_grounding_verifier(user_query: str, execution_metrics: dict, ltm: str, routing: IntentAnalysis) -> StructuredRAGResponse:
    critic_instruction = f"""You are 'Agent 3: Grounding Verifier & Critic'. Your job is to audit the raw text response draft compiled by Agent 2 against the raw baseline context data AND Long-Term Memory. You must structure the output into the final JSON response schema.

    STRICT AUDITING GROUND RULES:
    1. BOTH 'RAW CONTEXT DATA' AND 'LONG-TERM MEMORY ANCHORS' ARE VALID TRUTH SOURCES. If the user explicitly asks about their memory, role, preferences, or profile facts, and the answer is present in the LONG-TERM MEMORY ANCHORS or RAW CONTEXT DATA, it is considered 100% GROUNDED. Do NOT override it to 'Low' confidence.
    2. If 'IS DOCUMENT TARGETED' is True and the exact answer parameters are completely missing from BOTH the RAW CONTEXT DATA and LONG-TERM MEMORY, only then you MUST override Agent 2's draft. Set confidence_score to 'Low', source to 'Not Found'.
    3. NEVER allow Agent 2 to leak generic parametric knowledge or guess placeholders if no data exists in either source.

    RAW USER INPUT SENTENCE: "{user_query}"
    RAW CONTEXT DATA GATHERED BY AGENT 2: {execution_metrics['context']}
    LONG-TERM MEMORY ANCHORS: {ltm}
    AGENT 2 RAW RESPONSE DRAFT: {execution_metrics['draft']}
    IS DOCUMENT TARGETED MANDATE: {routing.is_document_targeted}
    EXPECTED SOURCE ROUTE: {execution_metrics['source']}
    """
    
    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=critic_instruction,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StructuredRAGResponse
        )
    )
    return StructuredRAGResponse.model_validate_json(response.text)

async def agent_4_output_inspector(user_query: str, verified_response: StructuredRAGResponse, execution_metrics: dict) -> StructuredRAGResponse:
    inspector_instruction = f"""You are 'Agent 4: Output Inspector Agent'. Your job is to conduct a final, strict quality control audit on the structured response generated by Agent 3.
    
    CRITICAL INSPECTION MANDATES:
    1. Verify that the response directly addresses the user's intent without breaking the structured JSON contract.
    2. Long-Term Memory & Tool Output Validation: If Agent 3 rejected an answer based on lack of PDF context, but the data IS present in context/memory, fix Agent 3's false-positive rejection.
    3. Formatting Polish: Return a fully audited and verified output matching the StructuredRAGResponse schema exactly.

    RAW USER QUERY: "{user_query}"
    RAW BASELINE CONTEXT DATA: {execution_metrics['context']}
    
    PROPOSED DATA FROM AGENT 3:
    - Answer: {verified_response.answer}
    - Summary Sentence: {verified_response.summary_sentence}
    - Confidence Score: {verified_response.confidence_score}
    - Source Path: {verified_response.source}
    
    Generate the final, verified, and polished structured response.
    """
    
    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=inspector_instruction,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StructuredRAGResponse
        )
    )
    return StructuredRAGResponse.model_validate_json(response.text)

# =====================================================================================
# 🌐 [HTTP ENDPOINTS]
# =====================================================================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html file not found!")
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/files")
async def list_files():
    if not os.path.exists("data"):
        return []
    return [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f)) and not f.startswith(".")]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not os.path.exists("data"):
        os.makedirs("data")
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    index_single_file(file_path, file.filename)
    return {"message": f"'{file.filename}' processed and incrementally indexed successfully!"}

@app.post("/ask")
async def ask_question(data: QueryModel):
    try:
        user_query = data.question

        # 🛡️ 1. Guardrail Step
        guardrail_result = run_input_guardrails(user_query)
        if not guardrail_result["safe"]:
            return {
                "answer": f"Güvenlik Protokolü Engellemesi: {guardrail_result['reason']}",
                "summary_sentence": "Kullanıcı girdisi güvenlik duvarı (Guardrail) tarafından engellendi.",
                "confidence_score": "Low",
                "source": "Input Guardrail Shield",
                "history": data.history
            }

        # 🧠 2. Memory Step
        await auto_extract_and_save_memory(user_id=data.user_id, user_msg=user_query)
        ltm_context = await get_long_term_memory(user_id=data.user_id, query=user_query)
        
        user_id_rule = """
        CRITICAL USER ID RESOLUTION RULE:
        - User IDs may be formatted with different spaces or underscores (e.g., 'user 101', 'user101', 'user_101').
        - You MUST treat 'user 101', 'user101', and 'user_101' as the EXACT SAME entity and ID. Spacing and writing style do not matter.
        """

        # 🔄 3. Sliding Window History Management (Last 6 entries)
        conversation = []
        recent_history = data.history[-6:] if data.history else []
        for msg in recent_history:
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

        conversation.append(types.Content(role="user", parts=[types.Part.from_text(text=user_query)]))
        
        # 🤖 4. Sequential Async Pipeline Execution
        history_summary_str = str(serialize_history(conversation[:-1]))
        
        routing_blueprint = await agent_1_intent_router(
            user_query=user_query, 
            history_summary=history_summary_str, 
            id_rule=user_id_rule
        )
        
        execution_results = await agent_2_data_executor(
            routing=routing_blueprint, 
            conversation_context=conversation, 
            ltm=ltm_context
        )
        
        interim_structured_response = await agent_3_grounding_verifier(
            user_query=user_query, 
            execution_metrics=execution_results, 
            ltm=ltm_context,
            routing=routing_blueprint
        )
        
        final_structured_response = await agent_4_output_inspector(
            user_query=user_query,
            verified_response=interim_structured_response,
            execution_metrics=execution_results
        )
        
        conversation.append({"role": "model", "parts": [{"text": final_structured_response.answer}]})
        
        return {
            "answer": final_structured_response.answer,
            "summary_sentence": final_structured_response.summary_sentence,
            "confidence_score": final_structured_response.confidence_score,
            "source": final_structured_response.source,
            "history": serialize_history(conversation)
        }

    except Exception as general_error:
        return {
            "answer": f"System processing failure encountered: {str(general_error)}",
            "summary_sentence": "An unexpected critical pipeline error occurred internally during multi-agent handling.",
            "confidence_score": "Low",
            "source": "System Error Safetynet",
            "history": []
        }