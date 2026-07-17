# Gemini Unified Copilot Backend

Enterprise-grade FastAPI backend integrated with the Google GenAI SDK (`gemini-3.1-flash-lite`), ChromaDB vector storage, and Docling parser. This copilot handles native tool execution, Local RAG, Autonomous Long-Term Memory (LTM), and Web Fallbacks under strict grounding guardrails.

## 🚀 Features

- **Unified Mode Architecture**: Multi-stage waterfall pipeline that gracefully routes queries through System Tools -> Local Document RAG -> Internal/Parametric Knowledge -> Live Web Search.
- **Short-Term Memory Preservation**: Full conversation state tracking passed seamlessly across the entire RAG pipeline without losing context between steps.
- **Autonomous Long-Term Memory (LTM)**: Asynchronous background extraction engine that catches, vectorizes, and saves permanent user facts/preferences into a dedicated ChromaDB collection (`user_long_term_memory`).
- **Strict Grounding Guardrails**: Zero-tolerance prompt structure designed to completely eliminate hallucinated information (e.g., placeholder identity injection) when document content is unavailable.
- **Advanced Document Parsing**: Integration with `Docling` for robust layout-aware Markdown extraction from PDFs and TXT files.

## 🛠️ Project Structure

- `rag.py`: Main FastAPI server containing routing, tool definitions, and Gemini lifecycle management.
- `index.html`: Responsive unified web chat interface.
- `chroma_data/`: Persistent vector store directory.
- `data/`: Target folder for uploaded and indexed workspace files.

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