import streamlit as st
import openai
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load your OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up GPT-4o model and embedding model
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding()

# Load documents from the 'docs' folder and index them
with st.spinner("Indexing school documents..."):
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

# Streamlit UI
st.title("📚 School GPT Assistant v0.4")
st.subheader("Ask me a question!")

user_input = st.text_input("")

if user_input:
    with st.spinner("Let me cook..."):
        response = query_engine.query(user_input)
        st.write(response.response)
