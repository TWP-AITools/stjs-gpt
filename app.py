import streamlit as st
import openai
import os, json, re, unicodedata, difflib
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Optional postprocessor (won't break if missing) ---
try:
    from llama_index.core.postprocessor import SimilarityPostprocessor
    HAVE_SIM_POST = True
except Exception:
    HAVE_SIM_POST = False

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

# ---------- Helpers: normalization & fuzzy matching ----------
def norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[\W_]+", " ", s)
    return " ".join(s.split())

def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=norm(a), b=norm(b)).ratio()

# ---------- Intent: only show downloads when explicitly requested ----------
def wants_download(user_q: str) -> bool:
    q = (user_q or "").lower()
    keywords = [
        "download", "attach", "attachment", "pdf", "file", "form",
        "link", "open the form", "printable", "save a copy", "give me the",
        "send me the", "provide the form", "share the form"
    ]
    return any(k in q for k in keywords)

# ---------- Manifest (optional but recommended) ----------
# Example manifest.json structure:
# [
#   {"path":"docs/SJCounselingReferralForm.pdf","title":"Counseling Referral Form","aliases":["counseling request","student counseling"],"keywords":["counsel","referral"]},
#   {"path":"docs/SpedReferral.pdf","title":"Special Education Referral Form","aliases":["sped referral","special ed referral"],"keywords":["idea","evaluation"]}
# ]
def load_manifest():
    try:
        with open("manifest.json","r",encoding="utf-8") as f:
            items = json.load(f)
    except FileNotFoundError:
        # create from files in /docs as a fallback
        items = []
        if os.path.isdir("docs"):
            for fn in sorted(os.listdir("docs")):
                p = os.path.join("docs", fn)
                if os.path.isfile(p) and p.lower().endswith((".pdf",".doc",".docx",".txt")):
                    title = os.path.splitext(os.path.basename(p))[0].replace("_"," ").replace("-"," ").title()
                    items.append({"path": p, "title": title, "aliases": [], "keywords": []})
    by_path = {it["path"]: it for it in items}
    flat = items
    return by_path, flat

MANIFEST_BY_PATH, MANIFEST = load_manifest()

def best_manifest_match(user_q: str):
    """Return (path, score, title) for the best direct/alias match, or (None,0,None)."""
    qn = norm(user_q)
    if not qn:
        return None, 0.0, None

    best = (None, 0.0, None)
    for item in MANIFEST:
        title = item.get("title") or os.path.basename(item["path"])
        cand_texts = [title] + item.get("aliases", []) + item.get("keywords", [])
        cand_texts.append(os.path.basename(item["path"]))  # filename
        exact_hit = any(norm(ct) in qn or qn in norm(ct) for ct in cand_texts if ct)
        score = 1.1 if exact_hit else max((fuzzy_ratio(user_q, ct) for ct in cand_texts if ct), default=0.0)
        if score > best[1]:
            best = (item["path"], score, title)
    return best

# ✅ Cached document loading/indexing
@st.cache_resource
def load_index():
    # Attach simple metadata (title + path) so sources are cleaner
    def file_meta(pathlike):
        p = str(pathlike).replace("\\","/")
        rec = MANIFEST_BY_PATH.get(p, {})
        return {
            "title": rec.get("title", os.path.basename(p)),
            "path": p,
            "file_name": os.path.basename(p),
        }

    documents = SimpleDirectoryReader(
        "docs",
        file_metadata=file_meta,
        required_exts=[".pdf",".docx",".doc",".txt"]
    ).load_data()

    index = VectorStoreIndex.from_documents(documents)

    # Build a query engine with tighter retrieval
    kwargs = dict(similarity_top_k=6)
    if HAVE_SIM_POST:
        kwargs["node_postprocessors"] = [SimilarityPostprocessor(similarity_cutoff=0.72)]
    return index.as_query_engine(**kwargs)

with st.spinner("Indexing school documents..."):
    query_engine = load_index()

# ✅ Setup session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito&display=swap');
    html, body, [class*="st-"] { font-family: 'Nunito', sans-serif; background-color: #121212; color: white; }
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #1e1e1e; color: white; border: 1px solid #228B22; border-radius: 5px; padding: 8px; resize: none;
    }
    .stTextArea > div > textarea { height: 100px !important; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }
    .stTextArea label, .stTextInput label { color: #228B22; }
    .stButton > button { background-color: #228B22 !important; color: white !important; border: none; border-radius: 5px; padding: 0.5em 1em; font-weight: bold; }
    .response-box { border: 1px solid #228B22; padding: 1rem; border-radius: 10px; background-color: #1e1e1e; color: white; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ✅ App Header
st.markdown("<h1 style='color:#228B22;'>🪓 St. John Public School Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:white;'>Hi, I'm <strong>Chucky</strong>. I'm your super smart, super serious school assistant. Ask me for a form or question and I will do my best to assist you!</p>", unsafe_allow_html=True)

# ✅ Helper to collect downloads from retrieval result
def collect_downloads(resp_obj, limit=3):
    files = []
    seen = set()
    for node in (resp_obj.source_nodes or [])[:limit]:
        meta = (node.metadata or {})
        path = (
            meta.get("file_path") or
            meta.get("path") or
            meta.get("source") or
            meta.get("filename") or
            ""
        )
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen or not os.path.exists(abs_path):
            continue
        seen.add(abs_path)
        files.append((abs_path, meta.get("title") or os.path.basename(abs_path)))
    return files

# ✅ Display chat history
for turn in st.session_state.chat_history:
    st.markdown(f"<div class='response-box'><strong>You:</strong> {turn['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='response-box'><strong>Chucky:</strong> {turn['bot']}</div>", unsafe_allow_html=True)

# ✅ User input and send button
user_input = st.text_area("Ask Chucky a Question:", key="user_input", height=100)

if st.button("Send") and user_input:
    with st.spinner("Let me cook..."):
        # ---- 1) Exact/alias match pass (manifest) ----
        direct_path, direct_score, direct_title = best_manifest_match(user_input)
        strong_direct = direct_path and (direct_score >= 1.0 or direct_score >= 0.88)

        # ---- 2) Semantic retrieval pass ----
        resp_obj = query_engine.query(user_input)
        context_text = resp_obj.response
        retrieved_files = collect_downloads(resp_obj, limit=5)

        # If direct match found but not in retrieved, prepend it
        if strong_direct:
            abs_direct = os.path.abspath(direct_path)
            if os.path.exists(abs_direct) and all(abs_direct != p for p,_ in retrieved_files):
                retrieved_files = [(abs_direct, direct_title)] + retrieved_files

        has_downloads = len(retrieved_files) > 0
        show_downloads = wants_download(user_input) and has_downloads

        # ---- 3) Build LLM answer ----
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, laid-back school assistant named Chad (aka Chucky). "
                    "Prefer exact document matches when the user names a form (title/filename/alias). "
                    "Only offer downloads if the user asks for a file/link; otherwise, answer concisely without attachments. "
                    "Do NOT claim you cannot provide files."
                )
            }
        ]
        if len(st.session_state.chat_history) >= 1:
            messages.append({"role": "user", "content": st.session_state.chat_history[-1]["user"]})
            messages.append({"role": "assistant", "content": st.session_state.chat_history[-1]["bot"]})

        direct_hint = f"\n\nLikely document: {direct_title}" if strong_direct and direct_title else ""
        messages.append({
            "role": "user",
            "content": f"The user asked: {user_input}{direct_hint}\n\nHere is the context I found in the documents:\n{context_text}"
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )
        answer = response.choices[0].message.content

        # Only append the download line if we're actually showing them
        if show_downloads:
            answer += "\n\n📎 You can download the relevant form(s) below."

        # ---- 4) Display answer ----
        st.session_state.chat_history.append({"user": user_input, "bot": answer})
        st.markdown(f"<div class='response-box'><strong>Chucky:</strong> {answer}</div>", unsafe_allow_html=True)

        # ---- 5) Render downloads only when explicitly requested ----
        if show_downloads:
            for abs_path, title in retrieved_files[:3]:
                st.caption(f"Source: {title}")
                mime = "application/pdf" if abs_path.lower().endswith(".pdf") else "application/octet-stream"
                with open(abs_path, "rb") as f:
                    st.download_button(
                        label="Download file",
                        data=f.read(),
                        file_name=os.path.basename(abs_path),
                        mime=mime
                    )