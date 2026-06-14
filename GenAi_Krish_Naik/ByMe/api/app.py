from fastapi import FastAPI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langserve import add_routes

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server using Gemini and Local Llama 3.2"
)

# Initialize models
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
local_llama = ChatOllama(model="llama3.2:3b")

# Define prompt templates
essay_prompt = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
poem_prompt = ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

# 1. Exposed Base Gemini Route
add_routes(
    app,
    gemini_model,
    path="/gemini"
)

# 2. Chain Route: Essay (Gemini)
add_routes(
    app,
    essay_prompt | gemini_model,
    path="/essay"
)

# 3. Chain Route: Poem (Local Llama 3.2:3b)
add_routes(
    app,
    poem_prompt | local_llama,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)