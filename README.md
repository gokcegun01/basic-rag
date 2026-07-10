# Gemini & Docling Powered Structured RAG API

This project is an enterprise-grade **Document-Based Question Answering (RAG) API service** built using **FastAPI**, **ChromaDB**, and the latest **Google GenAI SDK** (`gemini-3.1-flash-lite`, `gemini-embedding-2`). 

Unlike traditional PDF parsers, it integrates **IBM Docling** to flawlessly analyze and reconstruct complex document layouts and multi-column tables into clean Markdown format.

## 🚀 Key Features

* **Advanced PDF Parsing:** Powered by IBM Docling for high-fidelity document layout analysis and seamless table extraction into semantic Markdown.
* **Adaptive Rate-Limit & Token Optimization:** Implements an intelligent block-consolidation mechanism that merges consecutive paragraphs. This enriches the context window for Gemini while guaranteeing absolute immunity against API throttling and 429 RESOURCE_EXHAUSTED errors during high-volume document ingestion.
* **Strict Structured Output:** Enforces guaranteed JSON schemas (answer, summary_sentence, confidence_score) directly via Gemini's native response_schema configuration and Pydantic models to ensure seamless downstream predictability.
* **Persistent Vector Store:** Efficiently leverages ChromaDB for local vector storage, embedded document management, and ultra-fast semantic search.

## 🛠️ Installation & Setup

### Step 1: Install Dependencies
Run the following command in your terminal to install all required packages:
pip install -r requirements.txt

### Step 2: Configure Environment Variables
Create a file named ".env" in the root directory of your project and insert your Gemini API Key as follows:
API_KEY=your_google_gemini_api_key_here

### Step 3: Prepare Your Documents
Place any .pdf or .txt files you wish to index inside the "data/" directory. 
(Note: Whenever you update or change your source documents, remember to delete the local "chroma_data/" folder to trigger a fresh re-indexing).

### Step 4: Launch the API Server
Start your local Uvicorn development server with:
uvicorn rag:app --reload

### Step 5: Interactive API Testing
Open your browser and navigate to "http://127.0.0.1:8000/docs" to thoroughly test the /ask endpoint via the built-in interactive Swagger UI.