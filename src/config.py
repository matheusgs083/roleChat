import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do ambiente
load_dotenv()

# --- Configurações Gerais ---
APP_TITLE = "Pedro, o oraculo"  # Nome do assistente
PAGE_ICON = "🤖"

# --- Configurações do LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_ID = "llama3-70b-8192"
GROQ_TEMPERATURE = 0.3 
# --- Configurações do RAG ---

PDF_DIRECTORY = Path(".\\data\\pdfs")

FAISS_INDEX_PATH = "faiss_indexes"

# Use um embedding mais robusto para melhorar semântica
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

RETRIEVER_SEARCH_K = 5
RETRIEVER_FETCH_K = 10

INITIAL_AI_MESSAGE = f"Olá! Sou o assistente do {APP_TITLE}. Como posso te ajudar?"
