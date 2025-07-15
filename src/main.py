import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from rag_system import RAGSystem
import config as config

# Configurações da página
st.set_page_config(page_title=config.APP_TITLE, page_icon=config.PAGE_ICON)
st.title(config.APP_TITLE)

# Inicialização do sistema RAG (com cache)
@st.cache_resource
def get_rag_system_instance():
    return RAGSystem()

rag_system = get_rag_system_instance()

# Controle do histórico da conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content=config.INITIAL_AI_MESSAGE)
    ]

# Exibição do histórico
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# Entrada do usuário
user_input = st.chat_input("Digite sua pergunta aqui...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("Human"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        with st.spinner("Pensando..."):
            answer = rag_system.invoke_rag_chain(
                user_input,
                st.session_state.chat_history
            )

        st.session_state.chat_history.append(AIMessage(content=answer))
        st.markdown(answer)
