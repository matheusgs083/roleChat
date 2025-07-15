import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- General settings ---
APP_TITLE = "Pedro, o oraculo" # oracle name
PAGE_ICON = "🤖"

# ---  LLM settings (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_ID = "llama3-70b-8192"
GROQ_TEMPERATURE = 0.7

# --- RAG settings ---

PDF_DIRECTORY = Path(".\\data\\pdfs")
# Ajuste o caminho do índice FAISS para a nova estrutura de pastas (faiss_indexes/)
FAISS_INDEX_PATH = "faiss_indexes"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1" # Change to switch the embedding model

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_SEARCH_K = 3
RETRIEVER_FETCH_K = 4

# --- Initial message ---
INITIAL_AI_MESSAGE = f"Olá! Sou o assistente do {APP_TITLE}. Como posso te ajudar?"