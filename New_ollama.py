import io, uuid, hashlib, sqlite3
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxFile
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper

# SERPER_API_KEY ko environment variable me rakho (setx SERPER_API_KEY "key")
st.set_page_config(page_title="Ask Puneet", page_icon="🤖", layout="wide")
DB = "chat_history.db"

# ---------- History (SQLite) ----------
def _conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, role TEXT, content TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    return c

def save_msg(sid, role, content):
    with _conn() as c:
        c.execute("INSERT INTO messages(session_id, role, content) VALUES(?,?,?)",
                  (sid, role, content))

def load_msgs(sid):
    with _conn() as c:
        return c.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                         (sid,)).fetchall()

def list_sessions():
    with _conn() as c:
        return c.execute("""SELECT session_id,
            (SELECT content FROM messages m2 WHERE m2.session_id=m.session_id
             AND role='user' ORDER BY id LIMIT 1)
            FROM messages m GROUP BY session_id ORDER BY MAX(id) DESC""").fetchall()

# ---------- Documents ----------
@st.cache_resource
def get_store():
    return InMemoryVectorStore(OllamaEmbeddings(model="nomic-embed-text"))

def extract_text(f):
    name = f.name.lower()
    data = f.getvalue()
    if name.endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in DocxFile(io.BytesIO(data)).paragraphs)
    if name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        return "\n\n".join(f"Sheet {n}\n{df.to_csv(index=False)}" for n, df in sheets.items())
    return data.decode("utf-8", errors="ignore")  # txt, md, csv, json, py ...

def index_file(f):
    text = extract_text(f)
    if not text.strip():
        return 0
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_text(text)
    get_store().add_texts(chunks, metadatas=[{"source": f.name}] * len(chunks))
    return len(chunks)

# ---------- Agent ----------
@tool
def search_documents(query: str) -> str:
    """Search the user's uploaded documents (PDF, Word, Excel, text) for relevant passages."""
    docs = get_store().similarity_search(query, k=4)
    if not docs:
        return "No relevant content found in uploaded documents."
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

search = GoogleSerperAPIWrapper()

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return search.run(query)

@st.cache_resource
def get_agent():
    return create_agent(
        model=ChatOllama(model="gemma4:31b-cloud", temperature=0),
        tools=[search_documents, web_search],
        system_prompt=(
            "You are a helpful assistant. If the question may relate to uploaded files, "
            "use search_documents first and mention the file name. "
            "Use web_search for current information. "
            "Reply in the same language the user writes in."
        ),
    )

def answer(sid, text):
    history = [{"role": r, "content": c} for r, c in load_msgs(sid)][-20:]  # last 20 msgs
    save_msg(sid, "user", text)
    result = get_agent().invoke({"messages": history + [{"role": "user", "content": text}]})
    reply = result["messages"][-1].content
    if isinstance(reply, list):
        reply = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in reply)
    save_msg(sid, "assistant", reply)
    return reply

# ---------- Voice ----------
@st.cache_resource
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("small", compute_type="int8")  # Hindi + English auto-detect

def transcribe(audio_bytes):
    segments, _ = get_whisper().transcribe(io.BytesIO(audio_bytes))
    return " ".join(s.text.strip() for s in segments).strip()

# ---------- UI ----------
ss = st.session_state
ss.setdefault("sid", str(uuid.uuid4()))
ss.setdefault("indexed", set())
ss.setdefault("last_audio", None)

with st.sidebar:
    st.title("🤖 Ask Puneet")
    if st.button("➕ New chat", use_container_width=True):
        ss.sid = str(uuid.uuid4())
        st.rerun()
    st.caption("Purani chats")
    for sid, title in list_sessions():
        label = ("▶ " if sid == ss.sid else "") + (title or "Chat")[:35]
        if st.button(label, key=sid, use_container_width=True):
            ss.sid = sid
            st.rerun()

tab_chat, tab_voice, tab_files = st.tabs(["💬 Chat", "🎤 Voice", "📁 Files"])

with tab_files:
    files = st.file_uploader("Documents upload karo", accept_multiple_files=True,
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md", "json"])
    for f in files or []:
        key = (f.name, f.size)
        if key not in ss.indexed:
            with st.spinner(f"{f.name} padh raha hu..."):
                n = index_file(f)
            if n:
                ss.indexed.add(key)
                st.success(f"{f.name}: {n} parts index hue")
            else:
                st.warning(f"{f.name} me text nahi mila (scanned PDF ho sakta hai)")
    if ss.indexed:
        st.write("Ready:", ", ".join(n for n, _ in ss.indexed))

with tab_voice:
    audio = st.audio_input("Mic dabao aur bolo")
    if audio:
        raw = audio.getvalue()
        h = hashlib.md5(raw).hexdigest()
        if h != ss.last_audio:
            ss.last_audio = h
            with st.spinner("Awaaz text me badal raha hu..."):
                text = transcribe(raw)
            if text:
                st.info(f"Aapne kaha: {text}")
                ss.pending = text
                st.success("Jawab ke liye 💬 Chat tab kholo")
                st.rerun()
            else:
                st.warning("Awaaz samajh nahi aayi, dobara try karo.")

with tab_chat:
    for role, content in load_msgs(ss.sid):
        with st.chat_message(role):
            st.markdown(content)

# chat_input tab ke BAHAR hona chahiye, tabhi page ke neeche pin hota hai
typed = st.chat_input("Apna sawal likho...")
text = typed or ss.pop("pending", None)

with tab_chat:
    for role, content in load_msgs(ss.sid):
        with st.chat_message(role):
            st.markdown(content)

    if text:
        with st.chat_message("user"):
            st.markdown(text)
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hu..."):
                st.markdown(answer(ss.sid, text))