# AO RAG API

This project is a minimal, proof-of-concept implementation of a Retrieval-Augmented Generation (RAG) API using FastAPI. The API is designed to answer questions about cybersecurity KPIs based on the content of an Excel file.

## Features

- **FastAPI Backend:** A high-performance Python web framework for building APIs.
- **Retrieval-Augmented Generation (RAG):** A state-of-the-art NLP technique for question answering.
- **FAISS Indexing:** Efficient similarity search for retrieving relevant information.
- **Sentence Transformers:** Pre-trained models for generating high-quality text embeddings.

## Getting Started

### Prerequisites

- Python 3.7+
- Pip
- Virtualenv (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

```bash
uvicorn minimal_ao_api:app --reload
```

The API will be running at `http://127.0.0.1:8000`.

## API Usage

The API has a single endpoint for asking questions.

- **Endpoint:** `/ask`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
  	"query": "Your question about cybersecurity KPIs"
  }
  ```
- **Response Body:**
  ```json
  {
  	"answer": "The answer to your question."
  }
  ```

### Example with PowerShell

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/ask" -Body '{"query": "What is the KPI for phishing attacks?"}' -ContentType "application/json"
```
