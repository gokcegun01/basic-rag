# ⚡ OmniAssist AI | Enterprise RAG & Agentic Copilot

OmniAssist AI is a dual-mode, production-grade AI assistant that seamlessly fuses enterprise document management (RAG) with autonomous capabilities (Agentic Tool Calling). 

The system analyzes uploaded PDF/TXT documents, dynamically retrieves relevant context, and uses a **3-stage fallback pipeline** (Local RAG -> Parametric Memory -> Web Search) to answer queries. For system or database actions, it switches to **Agent Mode**, orchestrating internal Python functions via a secure **Manual Function Calling** loop.

---

## 🌟 Key Features

### 1. 📄 PDF & Web Chat (RAG Mode) - *Advanced Fallback Pipeline*
* **High-Fidelity PDF Parsing:** Leverages IBM's robust `Docling` engine to parse complex PDF and TXT layouts into clean Markdown before structural chunking.
* **Vector DB Integration:** Chunked documents are vectorized using the `gemini-embedding-2` model and indexed locally in a persistent `ChromaDB` instance.
* **3-Stage Smart Fallback Pipeline:** 
  1. **Local RAG:** The query is first searched within the local indexed documents.
  2. **Parametric Memory (LLM Core Knowledge):** If the answer is not found in the documents, the model utilizes Gemini's internal knowledge pool for general information queries, avoiding unnecessary external API latency.
  3. **Web Search Fallback:** For real-time or dynamic queries not covered in local docs, the system automatically triggers an underlying, API-free **DuckDuckGo Search** module.

### 2. 🛠️ Agent Tools (Agentic Mode)
Empowers the LLM to transition from a chat engine into an action executor using a secure **Manual Function Calling Loop**. The backend exposes 4 production-grade system tools:

* 🎟️ `get_coupon_discount`: Checks the validity and exact discount percentage of specific promotional codes (e.g., *OKAN20*, *YAZ50*).
* 📋 `get_all_active_coupons`: Fetches a complete list of currently active coupons stored in the system.
* 📂 `list_indexed_pdfs`: Scans the server-side `data/` directory to safely list indexed PDF files in a standardized dictionary format.
* 💾 `query_user_profile`: Queries a mock relational user database using robust input-matching.
  * *Input Tolerance:* Handles typos and minor formatting errors (e.g., matching `user101` or `USER-101` to the normalized ID `user_101`).

### 3. 🎨 Modern & Responsive UI
* **Tailwind-Powered Dashboard:** A clean, single-page interface styled with modern UI principles.
* **Dynamic Mode Controls:** Form inputs, file upload zones, and UI placeholders dynamically adapt depending on whether **RAG** or **Agent** mode is active.
* **State Management:** Includes a dedicated "Clear Chat" utility to flush conversation state on both frontend and backend instantly.

---

## 🚀 Quick Start

### 1. Prerequisite Setup
Clone this repository to your local machine and navigate into the project directory:

    git clone <your-repository-url>
    cd basic-rag

### 2. Configure Virtual Environment
Create and activate your Python virtual environment:

    python -m venv .venv-1
    source .venv-1/bin/activate

### 3. Install Dependencies
Install all required libraries (including `python-multipart`):

    pip install -r requirements.txt

### 4. Set Up Environment Variables
Create a `.env` file in the root folder of the project:

    API_KEY=your_google_gemini_api_key_here

### 5. Launch the Application
Run the FastAPI application with Uvicorn:

    uvicorn rag:app --reload

Open your browser and navigate to:
👉 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)