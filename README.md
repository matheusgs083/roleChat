# roleChat

roleChat is an AI-powered conversational assistant that leverages Large Language Models (LLMs) to read, understand, and learn from documents in order to provide accurate and context-aware responses.  

Originally designed to explore corporate roles and job functions, roleChat has evolved into a versatile chat system capable of supporting a wide range of knowledge discovery and interactive information retrieval tasks.

## Technologies Used

This project employs a robust technological stack to create a **RAG (Retrieval Augmented Generation) system**, combining **LLM (Large Language Model)** capabilities with information retrieval from documents.

* **Streamlit**: Used to build the **interactive and dynamic user interface**, allowing for easy interaction with the chatbot assistant.
* **LangChain**: An essential framework that orchestrates the entire RAG pipeline. It handles **LLM orchestration** for generating responses, manages **conversation history** for coherent replies, and builds **retrieval chains** to search for relevant document information.
* **Groq**: A platform that provides the **large language model (LLM)** `llama3-70b-8192` for the assistant's text comprehension and generation capabilities, noted for its efficiency and speed.
* **PyMuPDFLoader**: A tool for **efficient text extraction from PDF files**, fundamental for loading the documents that feed the RAG system.
* **RecursiveCharacterTextSplitter**: Responsible for **splitting documents into "chunks" (smaller pieces)**, optimizing them for processing and information retrieval.
* **HuggingFace Embeddings (sentence-transformers/all-mpnet-base-v2)**: A model used to **generate vectorial embeddings** of text chunks, transforming them into numerical representations that enable semantic search.
* **FAISS (Facebook AI Similarity Search)**: A library for **efficient similarity search in vectors**, used to store and query the **vector database** of chunks, ensuring quick retrieval of the most relevant information.
* **python-dotenv**: Manages **environment variables**, such as API keys, securely and in an organized manner.
