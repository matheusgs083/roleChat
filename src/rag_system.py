# src/rag_system.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

import config as config


class RAGSystem:
    def __init__(self):
        # Use métodos cacheados sem passar 'self'
        self._llm = self._load_llm()
        self._retriever = self._get_retriever()
        self._rag_chain = self._get_rag_chain()

    @st.cache_resource
    def _load_llm(_self):  # ← Corrigido aqui
        if not config.GROQ_API_KEY:
            st.error("GROQ_API_KEY is not set. Please check the .env file.")
            st.stop()
        return ChatGroq(
            model=config.GROQ_MODEL_ID,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=None,
            timeout=None,
            max_retries=3,
            api_key=config.GROQ_API_KEY
        )

    @st.cache_data
    def _extract_text_from_pdf(_self, file_path: Path) -> str:
        loader = PyMuPDFLoader(str(file_path))
        doc = loader.load()
        return "\n".join([page.page_content for page in doc])

    @st.cache_resource
    def _get_retriever(_self):  # Use _self pra não dar problema de cache
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        
        index_path_obj = Path(config.FAISS_INDEX_PATH)
        st.write(f"Verificando índice em: {index_path_obj.resolve()}")
        
        if index_path_obj.exists() and any(index_path_obj.iterdir()):
            st.info(f"Carregando índice FAISS existente em '{config.FAISS_INDEX_PATH}'...")
            vectorstore = FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            st.info(f"Nenhum índice encontrado. Criando novo índice FAISS a partir dos PDFs em '{config.PDF_DIRECTORY}'...")
            pdf_files = list(config.PDF_DIRECTORY.glob("*.pdf")) + list(config.PDF_DIRECTORY.glob("*.PDF"))
            st.write(f"PDFs encontrados: {[str(f) for f in pdf_files]}")
            if not pdf_files:
                st.error(f"Nenhum arquivo PDF encontrado na pasta: {config.PDF_DIRECTORY}")
                st.stop()
            
            documents_content = []
            for f in pdf_files:
                text = _self._extract_text_from_pdf(f)
                st.write(f"Extraído {len(text)} caracteres do arquivo {f.name}")
                documents_content.append(text)
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            chunks = []
            for doc_content in documents_content:
                split_chunks = splitter.split_text(doc_content)
                st.write(f"Gerados {len(split_chunks)} chunks do documento")
                chunks.extend(split_chunks)
            
            if not chunks:
                st.error("Nenhum chunk foi gerado dos PDFs. Verifique o conteúdo dos PDFs e as configurações do splitter.")
                st.stop()
            
            st.write(f"Total de chunks: {len(chunks)}. Criando vetor de embeddings...")
            
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            
            # Garante que a pasta existe antes de salvar
            index_path_obj.mkdir(parents=True, exist_ok=True)
            
            vectorstore.save_local(config.FAISS_INDEX_PATH)
            st.success(f"Índice FAISS criado e salvo em '{config.FAISS_INDEX_PATH}'!")
            st.write(f"Arquivos na pasta de índice: {[f.name for f in index_path_obj.iterdir()]}")
        
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVER_SEARCH_K,
                "fetch_k": config.RETRIEVER_FETCH_K
            }
        )

    @st.cache_resource
    def _get_rag_chain(_self):
        context_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Dada a conversa anterior e a nova pergunta, reescreva a pergunta para ser independente do histórico."),
            MessagesPlaceholder("chat_history"),
            ("human", "Pergunta: {input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=_self._llm,
            retriever=_self._retriever,
            prompt=context_q_prompt,
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"""Você é um assistente útil da {config.APP_TITLE}.
Use os pedaços de contexto abaixo para responder a pergunta do usuário.
Se não souber a resposta, diga que não sabe com certeza.
Responda em português e seja direto.
"""),
            MessagesPlaceholder("chat_history"),
            ("human", "Pergunta: {input}\n\nContexto: {context}"),
        ])

        qa_chain = create_stuff_documents_chain(_self._llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, qa_chain)

    def invoke_rag_chain(self, user_input: str, chat_history: list):
        response = self._rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        answer = response["answer"]
        return answer.split("</think>")[-1].strip() if "</think>" in answer else answer.strip()
