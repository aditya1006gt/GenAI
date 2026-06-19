import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
# Corrected import path for Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Suppress the Wikipedia WebBaseLoader warning
os.environ["USER_AGENT"] = "Streamlit_Langchain_App/1.0"
GROQ_API_KEY=os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Groq API with Langchain")
llm=ChatGroq(model="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY)

prompt=ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only. If you provide an answer that is not based on the context, you will be penalized. If you the answer is correct I'll reward you $100
Please provide most accurate response according to the context.
<context>
{context}
<context>
Question: {input}
"""
)

documents_chain=create_stuff_documents_chain(llm, prompt)
retriever=st.session_state.vector.as_retriever()
retrieval_chain=create_retrieval_chain(retriever, documents_chain)

prompt = st.text_input("Enter your question here")

if prompt:
    start=time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response)
    end=time.process_time()
    print(f"Time taken: {end - start} seconds")
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

            
