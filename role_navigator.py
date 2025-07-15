from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import getpass

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



id_model = "llama3-70b-8192"
temperature = 0.7

llm = load_llm(id_model, temperature)

prompt = ChatPromptTemplate.from_template("Como calcular se um cpf é valido, explique a lógica?")
chain = prompt | llm | StrOutputParser()
response = chain.invoke({})
print(response)