import streamlit as st
import openai
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
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
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Set up GPT-4o model and embedding model
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding()

# ✅ Load documents from 'docs' folder
with st.spinner("Indexing school documents..."):
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

# ✅ Nunito Font, Dark Mode, Forest Green, Logo Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Nunito', sans-serif;
        background-color: #121212;
        color: white;
    }
    .stTextArea > div > textarea {
        color: white;
        background-color: #1e1e1e;
    }
    .stTextArea label {
        color: #228B22;
    }
    .stButton > button {
        background-color: #228B22;
        color: white;
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

# ✅ App Header with Logo and Title
st.markdown("<h1 style='color:#228B22;'>🪓 St. John Public School Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:white;'>Hi, I'm <strong>Chad</strong> (but you can call me Chucky if you'd like!). I'm your super-serious, super-smart school assistant! Ask away — policies, referrals, handbooks, you name it!</p>", unsafe_allow_html=True)

# ✅ Multi-line input box for better UX
user_input = st.text_area("Ask Chad/Chucky a Question:", height=100)

# ✅ Smart document-aware inference
if user_input:
    with st.spinner("Let me cook..."):
        # Step 1: Retrieve matching content
        retrieved_response = query_engine.query(user_input)
        context = retrieved_response.response

        # Step 2: GPT-4o inference with fallback reasoning
        prompt = f"""
You are a helpful school assistant named Chucky. A user has asked the following question:

\"{user_input}\"

Here is some relevant information from school documents you have access to:

{context}

Using both the user's question and the document info above, give the best, most helpful answer you can. If the question is vague, infer what they probably mean and respond clearly.
"""

        full_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful school assistant who answers clearly and accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        answer = full_response.choices[0].message.content
        st.markdown(f"<div class='response-box'><strong>Chucky:</strong> {answer}</div>", unsafe_allow_html=True)


