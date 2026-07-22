TR

1. 🛠️ Sistem Araçları ve İndeksleme
serialize_history(conv_list): SDK nesnelerini (types.Content, types.Part, Pydantic modelleri) saf JSON yapılara dönüştürür.

get_coupon_discount(coupon_code): yaz50 (%50), okan20 (%20), staj100 (%100) kupon kodlarını kontrol eder.

get_all_active_coupons(): Aktif kupon listesini döner.

list_indexed_pdfs(): data/ klasöründeki dosyaları listeler.

query_user_profile(user_id): Personel veritabanını sorgular. user_101 (Gökçe Naz Gün) ve user_102 (Hakan Bey) profillerini içerir; girdi karakterlerini (_, -, boşluk) normalleştirir.

search_web_free(query): DuckDuckGo (DDGS) üzerinden en alakalı 3 canlı sonucu çeker.

index_single_file(file_path, file_name): Artımlı indeksleme yapar.

.pdf dosyalarını Docling ile Markdown'a çevirir.

Paragrafları 4'erli gruplar halinde birleştirip chunk oluşturur.

gemini-embedding-2 modeli ile vektörleştirip ./chroma_data altındaki rag_documents koleksiyonuna kaydeder.

2. 🛡️ Güvenlik (Guardrails) ve Otonom Hafıza (LTM)
run_input_guardrails(user_query):

\b\d{11,16}\b regex'i ile 11-16 haneli PII (T.C. Kimlik No, Kredi Kartı) verilerini engeller.

ignore previous instructions, system prompt, override all instructions gibi Prompt Injection kalıplarını yakalar.

save_to_long_term_memory(user_id, insight): Kullanıcıya dair bilgileri mükerrerlik kontrolü yaparak user_long_term_memory koleksiyonuna ekler.

get_long_term_memory(user_id, query, limit=3): Vektör araması ile kullanıcı sorusuyla alakalı geçmiş hafıza verilerini getirir.

auto_extract_and_save_memory(user_id, user_msg): gemini-3.1-flash-lite modelini kullanarak kullanıcı mesajından kalıcı bilgileri çıkarır ve otomatik olarak hafızaya kaydeder.

🤖 3. 4 Ajanlı Sıralı Boru Hattı
Agent 1: Intent Router (agent_1_intent_router)

Model: gemini-3.1-flash-lite

Görevi: Sorguyu ve geçmişi analiz eder, user_id_rule normalizasyonunu uygular ve rota planını belirler.

Agent 2: Data Executor (agent_2_data_executor)

Model: gemini-3.1-flash-lite

Görevi: Rota kararına göre veriyi (Tool, ChromaDB RAG, DuckDuckGo Web) toplar ve LTM ile birleştirerek ham bir taslak yanıt (draft text) hazırlar.

Agent 3: Grounding Verifier (agent_3_grounding_verifier)

Model: gemini-3.1-flash-lite

Görevi: Taslağı ham veri ve LTM ile karşılaştırır. Halüsinasyon kontrolü yapar ve güven skorunu belirler.

Agent 4: Output Inspector (agent_4_output_inspector)

Model: gemini-3.1-flash-lite

Görevi: Yanıt biçimini ve LTM/Tool kaynaklı bilgilerin doğruluğunu tescil eden son denetçidir.

🌐 4. FastAPI Endpoint'leri
GET /: index.html dosyasını sunar.

GET /files: data/ klasöründeki dosyaları listeler.

POST /upload: Dosyayı kaydeder ve index_single_file() ile artımlı olarak veritabanına ekler.

POST /ask:

run_input_guardrails kontrolünü çalıştırır.

auto_extract_and_save_memory ve get_long_term_memory hafıza adımını yürütür.

Sohbet geçmişinin son 6 mesajını (data.history[-6:]) kayan pencere (sliding window) olarak alır.

4 asenkron ajanı sırayla (Agent 1 ➔ Agent 2 ➔ Agent 3 ➔ Agent 4) çalıştırır.

Yanıtı ve güncellenmiş geçmişi JSON olarak döner.



ENG

1. 🛠️ System Tools & Document Indexing
serialize_history(conv_list): Converts Google GenAI SDK objects (types.Content, types.Part, Pydantic models) into pure JSON structures.

get_coupon_discount(coupon_code): Checks coupon codes yaz50 (50%), okan20 (20%), staj100 (100%).

get_all_active_coupons(): Returns active coupon list.

list_indexed_pdfs(): Lists files inside the data/ directory.

query_user_profile(user_id): Queries personnel database (user_101: Gökçe Naz Gün, user_102: Hakan Bey) with string normalization (_, -, spaces).

search_web_free(query): Retrieves top 3 live search results using DuckDuckGo (DDGS).

index_single_file(file_path, file_name): Incremental indexing helper.

Converts .pdf files into Markdown via Docling.

Merges paragraphs in groups of 4 to create text chunks.

Embeds chunks via gemini-embedding-2 and stores them in the rag_documents collection under ./chroma_data.

2. 🛡️ Guardrails & Autonomous Long-Term Memory (LTM)
run_input_guardrails(user_query):

Blocks 11-16 digit PII sequences via \b\d{11,16}\b regex.

Detects Prompt Injection phrases (ignore previous instructions, system prompt, override all instructions).

save_to_long_term_memory(user_id, insight): Saves user facts into the user_long_term_memory collection with deduplication checks.

get_long_term_memory(user_id, query, limit=3): Retrieves semantically relevant long-term memory records via vector search.

auto_extract_and_save_memory(user_id, user_msg): Extracts permanent user facts using gemini-3.1-flash-lite in the background and stores them automatically.

🤖 3. 4-Agent Sequential Pipeline
Agent 1: Intent Router (agent_1_intent_router)

Model: gemini-3.1-flash-lite

Role: Analyzes query and history, applies user_id_rule normalization, and generates the execution blueprint.

Agent 2: Data Executor (agent_2_data_executor)

Model: gemini-3.1-flash-lite

Role: Retrieves data based on routing (Tools, ChromaDB RAG, DuckDuckGo Web), merges LTM context, and compiles a raw response draft.

Agent 3: Grounding Verifier (agent_3_grounding_verifier)

Model: gemini-3.1-flash-lite

Role: Audits draft text against raw context data and LTM facts. Checks for hallucinations and assigns confidence scores.

Agent 4: Output Inspector (agent_4_output_inspector)

Model: gemini-3.1-flash-lite

Role: Final quality auditor verifying response formatting and correcting false-positive rejections.

🌐 4. FastAPI Endpoints
GET /: Serves index.html.

GET /files: Lists files in the data/ folder.

POST /upload: Saves uploaded file and indexes it incrementally via index_single_file().

POST /ask:

Runs run_input_guardrails.

Executes auto_extract_and_save_memory and get_long_term_memory.

Applies sliding window history management on the last 6 messages (data.history[-6:]).

Runs 4 async agents sequentially (Agent 1 ➔ Agent 2 ➔ Agent 3 ➔ Agent 4).

Returns structured response payload and updated conversation history.