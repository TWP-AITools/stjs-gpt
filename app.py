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

# ✅ Load OpenAI API key
load_dotenv()
client = openai.OpenAI()
Settings.llm = LlamaOpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding()

# ✅ Cached document loading/indexing
@st.cache_resource
def load_index():
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

with st.spinner("Indexing school documents..."):
    query_engine = load_index()

# ✅ Setup session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Nunito', sans-serif;
        background-color: #121212;
        color: white;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #228B22;
        border-radius: 5px;
        padding: 8px;
        resize: none;
    }
    .stTextArea > div > textarea {
        height: 100px !important;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .stTextArea label, .stTextInput label {
        color: #228B22;
    }
    .stButton > button {
        background-color: #228B22 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .response-box {
        border: 1px solid #228B22;
        padding: 1rem;
        border-radius: 10px;
        background-color: #1e1e1e;
        color: white;
        margin-top: 1rem;
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

# ✅ User input and send button
user_input = st.text_area("Ask Chad/Chucky a Question:", key="user_input", height=100)
if st.button("Send") and user_input:
    with st.spinner("Let me cook..."):
        # Keep full response object so we can grab sources
        resp_obj = query_engine.query(user_input)
        context_text = resp_obj.response

        # Build chat messages
        messages = [
            {"role": "system", "content": "You are a helpful, laid-back school assistant named Chad (aka Chucky). Use the context provided to answer questions clearly and informally."}
        ]
        if len(st.session_state.chat_history) >= 1:
            messages.append({"role": "user", "content": st.session_state.chat_history[-1]["user"]})
            messages.append({"role": "assistant", "content": st.session_state.chat_history[-1]["bot"]})

        messages.append({
            "role": "user",
            "content": f"The user asked: {user_input}\n\nHere is the context I found in the documents:\n{context_text}"
        })

        # Get LLM answer
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.4
        )
        answer = response.choices[0].message.content

        # Save & display answer
        st.session_state.chat_history.append({"user": user_input, "bot": answer})
        st.markdown(f"<div class='response-box'><strong>Chucky:</strong> {answer}</div>", unsafe_allow_html=True)

        # Offer downloads for top 2–3 source docs
        try:
            shown = set()
            for node in (resp_obj.source_nodes or [])[:3]:
                meta = (node.metadata or {})

                # Try common metadata keys for file path
                path = (
                    meta.get("file_path") or
                    meta.get("path") or
                    meta.get("source") or
                    meta.get("filename") or
                    ""
                )
                title = (
                    meta.get("file_name") or
                    os.path.basename(path) or
                    meta.get("title") or
                    "Document"
                )

                if not path:
                    continue
                abs_path = os.path.abspath(path)
                if abs_path in shown or not os.path.exists(abs_path):
                    continue
                shown.add(abs_path)

                st.caption(f"Source: {title}")

                mime = "application/pdf" if abs_path.lower().endswith(".pdf") else "application/octet-stream"
                with open(abs_path, "rb") as f:
                    st.download_button(
                        "Download file",
                        data=f.read(),
                        file_name=os.path.basename(abs_path),
                        mime=mime
                    )
        except Exception as e:
            st.caption(f"Sources unavailable: {e}")