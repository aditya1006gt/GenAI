from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
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
st.title("Chatbot with LangChain, Streamlit and Gemini API")
input_text=st.text_input("Ask a question about the world:")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Create the output parser
output_parser = StrOutputParser()

# Chain them together (this is the LangChain 'chain')
chain = prompt | llm | output_parser
# isko hmlog chain kr rahe hain, pehle prompt se question lega, 
# fir llm se answer generate karega, 
# fir output parser se us answer ko parse karega

if input_text:
    st.write(chain.invoke({"question": input_text}))

