import streamlit as st
import openai
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 🔐 Password Protection Setup
PASSWORD = "Woodchucks2025!"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "tried_password" not in st.session_state:
    st.session_state["tried_password"] = False

if not st.session_state["authenticated"]:
    pw = st.text_input("Enter the magic word to access St. John Public School Assistant:", type="password")
    if pw:
        st.session_state["tried_password"] = True
        if pw == PASSWORD:
            st.session_state["authenticated"] = True
            st.success("✅ Open Sesame! Welcome!")
        else:
            st.error("⛔ Nuh Uh. Try Again!")
            st.stop()
    elif st.session_state["tried_password"]:
        st.error("⛔ Nuh Uh. Try Again!")
        st.stop()
    else:
        st.stop()

# ✅ Load OpenAI API key and configure models
load_dotenv()
client = openai.OpenAI()
Settings.llm = LlamaOpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding()

# ✅ Load documents from 'docs' folder and index them
with st.spinner("Indexing school documents..."):
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

# ✅ Setup session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Custom CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Nunito', sans-serif;
        background-color: #121212;
        color: white;
    }

    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #228B22;
        border-radius: 5px;
        padding: 8px;
    }

    .stTextArea > div > textarea {
        background-color: #1e1e1e !important;
        color: white !important;
        border: 1px solid #228B22 !important;
        border-radius: 5px !important;
        height: 100px !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        resize: none !important;
        padding: 8px;
    }

    .stTextArea label {
        color: #228B22;
    }

    .response-box {
        border: 1px solid #228B22;
        padding: 1rem;
        border-radius: 10px;
        background-color: #1e1e1e;
        color: white;
        margin-top: 1rem;
    }

    /* Remove Ctrl+Enter hint */
    [data-baseweb="textarea"] + div {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ App Header
st.markdown("<h1 style='color:#228B22;'>🪓 St. John Public School Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:white;'>Hi, I'm <strong>Chad</strong> (aka Chucky). I'm your super-serious, super-smart school assistant. Ask me about forms, standards, procedures, or anything else Chucks!</p>", unsafe_allow_html=True)

# ✅ Display chat history
for turn in st.session_state.chat_history:
    st.markdown(f"<div class='response-box'><strong>You:</strong> {turn['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='response-box'><strong>Chucky:</strong> {turn['bot']}</div>", unsafe_allow_html=True)

# ✅ User input field (Enter to submit)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Ask Chad/Chucky a Question:", key="user_input", height=100)
    submitted = st.form_submit_button("")

if submitted and user_input:
    with st.spinner("Let me cook..."):
        doc_response = query_engine.query(user_input).response

        messages = [
            {"role": "system", "content": "You are a helpful, laid-back school assistant named Chad (aka Chucky). Use the context provided to answer questions clearly and informally."}
        ]

        # 1-turn memory
        if len(st.session_state.chat_history) >= 1:
            messages.append({"role": "user", "content": st.session_state.chat_history[-1]["user"]})
            messages.append({"role": "assistant", "content": st.session_state.chat_history[-1]["bot"]})

        messages.append({
            "role": "user",
            "content": f"The user asked: {user_input}\n\nHere is the context I found in the documents:\n{doc_response}"
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.4
        )
        answer = response.choices[0].message.content

        st.session_state.chat_history.append({"user": user_input, "bot": answer})
        st.markdown(f"<div class='response-box'><strong>Chucky:</strong> {answer}</div>", unsafe_allow_html=True)
