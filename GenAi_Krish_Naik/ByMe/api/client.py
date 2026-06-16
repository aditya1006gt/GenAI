import requests
import streamlit as st

def get_gemini_response(question):
    response = requests.post("http://localhost:8000/essay/invoke", 
    json={"input": {"topic":question}})

    return response.json()["output"]["content"]

def get_ollama_response(question):
    response = requests.post("http://localhost:8000/poem/invoke", 
    json={"input": {"topic":question}})

    return response.json()["output"]["content"]

st.title("Langchain API Client")
input_text=st.text_input("Write an essay about the world:")
input_text2=st.text_input("Write a poem about the world for a 5 year old:")

if input_text:
    st.write(get_gemini_response(input_text))

if input_text2:
    st.write(get_ollama_response(input_text2))