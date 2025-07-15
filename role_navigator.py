from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from IPython.display import Markdown, display
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

def load_llm(id_model, temperature):
    return ChatGroq(
        model=id_model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=3
    )

def show_reply(res):
    try:
        res = str(res).strip()
        if "</think>" in res:
            res = res.split("</think>")[-1].strip()
        if "**" not in res and "*" not in res:
            if ":" in res:
                parts = res.split(":", 1)
                res = f"**{parts[0].strip()}:** {parts[1].strip()}"
        display(print(res)) ## alterar para Markdown quando estiver no Jupyter Notebook
    except Exception as e:
        print(f"[Error displaying reply]: {e}")

id_model = "llama3-70b-8192"
temperature = 0.7

llm = load_llm(id_model, temperature)

context = """
Para alterar uma senha no aplicativo, clique no menu 'Minha conta' e selecione 'Alterar senha'.
Para alterar a senha pelo site, acesse 'Configurações' no menu do topo. Em seguida, selecione 'Minha conta' e 'Alterar senha'.
"""

template_rag = """
Ask: {input}
Context: {context}
"""

prompt_rag = PromptTemplate.from_template(template_rag)

chain_rag = prompt_rag | llm | StrOutputParser()

input_text = "como alterar minha senha?"

res = chain_rag.invoke({"context": context, "input": input_text})

show_reply(res)
