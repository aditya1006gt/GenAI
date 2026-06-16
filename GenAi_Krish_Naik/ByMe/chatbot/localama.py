from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant that helps answer questions about the world."),
        ("user","Questions: {question}")
    ]
)


#  Streamlit Framework
st.title("Chatbot with LangChain, Streamlit and Local Llama 3.2")
input_text=st.text_input("Ask a question about the world:")


llm = ChatOllama(model="llama3.2:3b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

    