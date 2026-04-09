import streamlit as st
import os
from dotenv import load_dotenv
from repo_manager import RepoManager
from rag_engine import RAGEngine

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Codebase Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    .stApp { background-color: #ffffff; }
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu, footer, .stDeployButton, [data-testid="stDecoration"],
    [data-testid="stToolbar"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }

    /* ── Top bar area ── */
    .topbar {
        background: #111827;
        padding: 0.6rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0 -1rem;
        border-bottom: 1px solid #1f2937;
    }
    .topbar-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .topbar-logo {
        color: #fff;
        font-weight: 700;
        font-size: 15px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .topbar-sep { color: #374151; font-size: 18px; }
    .topbar-pill {
        background: rgba(129,140,248,0.12);
        border: 1px solid rgba(129,140,248,0.2);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        color: #a5b4fc;
        font-family: ui-monospace, 'SF Mono', 'Fira Code', monospace;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .topbar-pill .dot {
        width: 6px; height: 6px;
        background: #34d399;
        border-radius: 50%;
        display: inline-block;
    }
    .topbar-pill .dim { color: #6b7280; font-size: 11px; }

    /* ── Streamlit controls row (right under topbar) ── */
    .controls-row {
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        margin: 0 -1rem;
        padding: 0;
    }

    /* ── Popover buttons in controls row ── */
    [data-testid="stPopover"] > button {
        background: transparent !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        color: #374151 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 4px 12px !important;
    }
    [data-testid="stPopover"] > button:hover {
        background: #f3f4f6 !important;
        border-color: #d1d5db !important;
    }

    /* ── Action buttons ── */
    .stButton > button {
        background: transparent !important;
        color: #6b7280 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 4px 12px !important;
    }
    .stButton > button:hover {
        background: #f3f4f6 !important;
        color: #111827 !important;
        border-color: #d1d5db !important;
    }

    /* ── Form fields ── */
    [data-testid="stTextInput"] input {
        background-color: #fff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
        color: #111827 !important;
        font-size: 14px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(129,140,248,0.12) !important;
    }
    [data-testid="stTextInput"] label,
    .stMultiSelect label,
    .stSelectbox label,
    .stSlider label {
        color: #374151 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    .stMultiSelect [data-baseweb="select"] {
        background-color: #fff !important;
        border-color: #d1d5db !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #eef2ff !important;
        color: #4f46e5 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #fff !important;
        border-color: #d1d5db !important;
        color: #111827 !important;
    }

    /* ── Setup page button ── */
    .setup-btn > button {
        background: #111827 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 15px !important;
    }
    .setup-btn > button:hover { background: #1f2937 !important; }

    /* ── Popover inner button ── */
    [data-testid="stPopover"] .stButton > button {
        background: #111827 !important;
        color: #fff !important;
        border: none !important;
    }
    [data-testid="stPopover"] .stButton > button:hover {
        background: #1f2937 !important;
    }

    /* ── Status ── */
    [data-testid="stStatusWidget"] {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }

    /* ── Chat input ── */
    [data-testid="stBottom"] {
        background: #fff !important;
        border-top: 1px solid #f0f0f0;
    }
    [data-testid="stChatInput"] {
        background-color: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        max-width: 780px;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] textarea { color: #111827 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #9ca3af !important; }
    [data-testid="stChatInput"] button {
        background: #111827 !important;
        color: #fff !important;
        border-radius: 8px !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        max-width: 780px;
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] strong {
        color: #111827 !important;
    }
    [data-testid="stChatMessage"] code {
        color: #be185d !important;
        background: #fdf2f8 !important;
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 13px;
    }
    [data-testid="stChatMessage"] pre {
        background: #111827 !important;
        border-radius: 8px !important;
        border: none !important;
    }
    [data-testid="stChatMessage"] pre code {
        color: #e5e7eb !important;
        background: transparent !important;
    }

    /* ── Source expander ── */
    [data-testid="stExpander"] {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #6b7280 !important; }

    /* ── Setup page ── */
    .setup-wrap {
        max-width: 480px;
        margin: 12vh auto 0;
        text-align: center;
    }
    .setup-wrap .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .setup-wrap h1 {
        color: #111827;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .setup-wrap .sub { color: #6b7280; font-size: 0.92rem; margin-bottom: 2rem; }

    /* ── Empty chat ── */
    .empty-chat { text-align: center; margin-top: 15vh; }
    .empty-chat .icon { font-size: 2rem; margin-bottom: 0.4rem; }
    .empty-chat p { color: #9ca3af; font-size: 0.95rem; }
    .empty-chat .suggestions {
        display: flex; gap: 8px; justify-content: center;
        flex-wrap: wrap; margin-top: 1.2rem;
    }
    .empty-chat .sug {
        background: #f9fafb; border: 1px solid #e5e7eb;
        border-radius: 8px; padding: 8px 16px;
        font-size: 13px; color: #374151;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] [role="slider"] { background: #111827 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repos" not in st.session_state:
    st.session_state.repos = {}
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "repo_manager" not in st.session_state:
    st.session_state.repo_manager = RepoManager()
if "top_k" not in st.session_state:
    st.session_state.top_k = 8
if "search_type" not in st.session_state:
    st.session_state.search_type = "mmr"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2

repo_mgr = st.session_state.repo_manager
has_repos = len(st.session_state.repos) > 0


# ═════════════════════════════════════════════════════════════════════════════
#  SETUP SCREEN
# ═════════════════════════════════════════════════════════════════════════════
if not has_repos:
    st.markdown("""
    <div class="setup-wrap">
        <div class="icon">🧠</div>
        <h1>Codebase Assistant</h1>
        <p class="sub">Connect a Git repository to start exploring your code with AI.</p>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        repo_url = st.text_input("Repository URL", placeholder="https://github.com/user/repo")
        c1, c2 = st.columns(2)
        with c1:
            repo_branch = st.text_input("Branch", value="main")
        with c2:
            file_types = st.multiselect(
                "File types",
                [".py", ".js", ".jsx", ".tsx", ".ts", ".md", ".json", ".yaml", ".css", ".html", ".java", ".go", ".rs"],
                default=[".py", ".js", ".jsx", ".tsx", ".ts", ".md", ".json"],
            )

        st.markdown("")
        st.markdown('<div class="setup-btn">', unsafe_allow_html=True)
        if st.button("Clone & Index →", use_container_width=True):
            if not repo_url:
                st.error("Enter a repository URL.")
            else:
                repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
                try:
                    with st.status(f"Indexing {repo_name}…", expanded=True) as status:
                        st.write("📥 Cloning…")
                        docs = repo_mgr.clone_and_load(repo_url, repo_branch, file_types)
                        st.write(f"📄 Loaded **{len(docs)}** files")
                        st.write("✂️ Chunking…")
                        chunks = repo_mgr.split_documents(docs)
                        st.write(f"🧩 **{len(chunks)}** chunks")
                        st.write("🔍 Embedding…")
                        repo_mgr.index_chunks(chunks, repo_name)
                        st.session_state.repos[repo_name] = {
                            "url": repo_url, "branch": repo_branch,
                            "files": len(docs), "chunks": len(chunks),
                        }
                        st.session_state.rag_engine = RAGEngine(repo_mgr)
                        status.update(label=f"✅ {repo_name} indexed", state="complete")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  CHAT SCREEN
# ═════════════════════════════════════════════════════════════════════════════

# ── Top bar (HTML for branding) ──────────────────────────────────────────────
pills = ""
for name, info in st.session_state.repos.items():
    pills += f'<span class="topbar-pill"><span class="dot"></span>{name} <span class="dim">· {info["files"]} files · {info["chunks"]} chunks</span></span> '

st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <span class="topbar-logo">🧠 Codebase Assistant</span>
        <span class="topbar-sep">|</span>
        {pills}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Controls row ─────────────────────────────────────────────────────────────
if "advanced_unlocked" not in st.session_state:
    st.session_state.advanced_unlocked = False

c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.2, 1.2, 1.6, 1, 1])
with c1:
    with st.popover("➕ Add repo"):
        add_url = st.text_input("URL", placeholder="https://github.com/…", key="add_url")
        add_branch = st.text_input("Branch", value="main", key="add_branch")
        add_types = st.multiselect(
            "Types", [".py", ".js", ".jsx", ".tsx", ".ts", ".md", ".json"],
            default=[".py", ".js", ".jsx", ".tsx", ".ts", ".md", ".json"], key="add_types",
        )
        if st.button("Index", use_container_width=True, key="add_btn"):
            if add_url:
                rn = add_url.rstrip("/").split("/")[-1].replace(".git", "")
                try:
                    with st.spinner(f"Indexing {rn}…"):
                        docs = repo_mgr.clone_and_load(add_url, add_branch, add_types)
                        chunks = repo_mgr.split_documents(docs)
                        repo_mgr.index_chunks(chunks, rn)
                        st.session_state.repos[rn] = {
                            "url": add_url, "branch": add_branch,
                            "files": len(docs), "chunks": len(chunks),
                        }
                        st.session_state.rag_engine = RAGEngine(repo_mgr)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
with c2:
    with st.popover("🔒 Advanced"):
        if not st.session_state.advanced_unlocked:
            st.caption("Enter passcode to unlock advanced settings")
            code = st.text_input("Passcode", type="password", key="adv_code")
            if st.button("Unlock", use_container_width=True, key="adv_unlock"):
                if code == os.getenv("ADVANCED_CODE", "admin123"):
                    st.session_state.advanced_unlocked = True
                    st.rerun()
                else:
                    st.error("Wrong passcode")
        else:
            st.caption("🔓 Advanced settings unlocked")
            st.session_state.top_k = st.slider("Top-K chunks", 2, 20, st.session_state.top_k, key="sk")
            st.session_state.search_type = st.selectbox(
                "Search strategy", ["mmr", "similarity"],
                index=0 if st.session_state.search_type == "mmr" else 1, key="ss"
            )
            st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, step=0.1, key="st_")
            if st.button("🔒 Lock", use_container_width=True, key="adv_lock"):
                st.session_state.advanced_unlocked = False
                st.rerun()
with c3:
    if st.button("🧹 Clear", key="cc"):
        st.session_state.messages = []
        st.rerun()
with c6:
    if st.button("🗑️ Reset all", key="ra"):
        repo_mgr.clear_all()
        st.session_state.repos = {}
        st.session_state.rag_engine = None
        st.session_state.messages = []
        st.rerun()

st.markdown('<div style="border-bottom: 1px solid #e5e7eb; margin: 0 -1rem;"></div>', unsafe_allow_html=True)

# ── RAG engine ───────────────────────────────────────────────────────────────
if st.session_state.rag_engine is None:
    st.session_state.rag_engine = RAGEngine(repo_mgr)
rag = st.session_state.rag_engine

# ── Empty state ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-chat">
        <div class="icon">💬</div>
        <p>Ask anything about your codebase</p>
        <div class="suggestions">
            <span class="sug">How is the project structured?</span>
            <span class="sug">Explain the auth flow</span>
            <span class="sug">How do I set up locally?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source files"):
                for src in msg["sources"]:
                    st.caption(f"`{src['file']}`")
                    st.code(src["snippet"], language=src.get("lang", ""))

# ── Chat input ───────────────────────────────────────────────────────────────
if query := st.chat_input("Ask about the codebase…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history = st.session_state.messages[-12:]
    chat_history = [
        ("human" if m["role"] == "user" else "assistant", m["content"])
        for m in history
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                result = rag.ask(
                    query,
                    chat_history=chat_history,
                    top_k=st.session_state.top_k,
                    search_type=st.session_state.search_type,
                    temperature=st.session_state.temperature,
                )
            st.markdown(result["answer"])

            sources = result.get("sources", [])
            if sources:
                with st.expander(f"📎 {len(sources)} source files"):
                    for src in sources:
                        st.caption(f"`{src['file']}`")
                        st.code(src["snippet"], language=src.get("lang", ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": sources,
            })
        except Exception as e:
            error_msg = f"Sorry, something went wrong: {e}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
            })