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
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("A chave da API GROQ não foi encontrada. Certifique-se de que está definida no arquivo .env.")
    st.stop()

st.set_page_config(page_title="Atendimento Pau Brasil🤖", page_icon="🤖")
st.title("Atendimento Pau Brasil")

ID_MODEL = "llama3-70b-8192"
TEMPERATURE = 0.7
PDF_DIR = Path("C:\\Users\\caixa.patos\\Downloads\\Notas")
FAISS_INDEX_PATH = "index_faiss"
# ALTO IMPACTO: Novo modelo de embedding
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

@st.cache_resource
def load_llm(model_id: str, temp: float):
    return ChatGroq(
        model=model_id,
        temperature=temp,
        max_tokens=None,
        timeout=None,
        max_retries=3,
    )

@st.cache_data
def extract_text_from_pdf(file_path: Path) -> str:
    loader = PyMuPDFLoader(str(file_path))
    doc = loader.load()
    return "\n".join([page.page_content for page in doc])

@st.cache_resource
def get_retriever(folder_path: Path, index_path: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(index_path).exists():
        st.info("Carregando índice FAISS existente...")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        st.info("Criando novo índice FAISS a partir dos PDFs...")
        pdf_files = list(folder_path.glob("*.pdf"))
        if not pdf_files:
            st.error(f"Nenhum PDF encontrado na pasta: {folder_path}")
            st.stop()

        documents_content = [extract_text_from_pdf(f) for f in pdf_files]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for doc_content in documents_content:
            chunks.extend(splitter.split_text(doc_content))

        if not chunks:
            st.error("Nenhum chunk foi gerado a partir dos PDFs. Verifique o conteúdo dos PDFs ou a configuração do splitter.")
            st.stop()

        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        vectorstore.save_local(index_path)
        st.success("Índice FAISS criado e salvo com sucesso!")

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 4}
    )

@st.cache_resource
def get_rag_chain(_llm_model: ChatGroq, _retriever_obj) -> callable:
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Dada a conversa anterior e a nova pergunta, reescreva a pergunta para ser independente do histórico."),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=_llm_model,
        retriever=_retriever_obj,
        prompt=context_q_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Você é um assistente útil da Distribuidora Pau Brasil Ambev.
Use os pedaços de contexto abaixo para responder a pergunta do usuário.
Se não souber a resposta, diga que não sabe com certeza.
Responda em português e seja direto.
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}\n\nContexto: {context}"),
    ])

    qa_chain = create_stuff_documents_chain(_llm_model, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)

llm = load_llm(ID_MODEL, TEMPERATURE)
retriever = get_retriever(PDF_DIR, FAISS_INDEX_PATH)
rag_chain = get_rag_chain(llm, retriever)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá! Sou o assistente da Pau Brasil. Como posso te ajudar?")
    ]

for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

user_input = st.chat_input("Digite sua pergunta aqui...")

if user_input:
    with st.chat_message("Human"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history
        })

        answer = response["answer"]
        answer = answer.split("</think>")[-1].strip() if "</think>" in answer else answer.strip()

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))
        st.markdown(answer)