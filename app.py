import os
import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🏢", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
.sub-header { color: #666; font-size: 1rem; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── API Key — hardcoded ──
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in secrets!")
    st.stop()

# ── Session state ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = []

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_rag_chain(vectorstore):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY,
    )
    RAG_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an Enterprise Knowledge Assistant. Answer using ONLY the context below.
If not found say: I don't have that information in the knowledge base.
Always cite the source document at the end.

Context:
{context}

Question: {question}

Answer:"""
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    def format_docs(docs):
        return "\n\n".join([
            f"[{d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
            for d in docs
        ])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain, retriever

def load_document(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loaders = {"pdf": PyPDFLoader, "txt": TextLoader, "csv": CSVLoader}
    if ext not in loaders:
        return []
    docs = loaders[ext](tmp_path).load()
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name
    os.unlink(tmp_path)
    return docs

SAMPLE_DOCS = [
    Document(page_content="""HR POLICY - LEAVE MANAGEMENT
Annual Leave: 18 days paid leave per year.
Sick Leave: 12 days per year. Medical certificate needed after 3 days.
Maternity Leave: 26 weeks paid. Paternity Leave: 15 days paid.
Leave must be applied 7 days in advance via HR portal.""",
    metadata={"source": "HR_Policy.pdf"}),
    Document(page_content="""IT SECURITY POLICY
Passwords: Minimum 12 characters, uppercase, lowercase, numbers, symbols.
Rotation: Every 90 days mandatory.
VPN: Required for all remote work.
Breach Reporting: Report to IT helpdesk within 1 hour.""",
    metadata={"source": "IT_Security.pdf"}),
    Document(page_content="""ONBOARDING GUIDE
Week 1: Identity verification, laptop, orientation.
Week 2: Compliance training on LMS.
Week 3: Shadow team lead, tool training.
Week 4: First project, 1-on-1 with manager.
Enroll in health insurance within 30 days.""",
    metadata={"source": "Onboarding.pdf"}),
    Document(page_content="""EXPENSE POLICY
Domestic travel: Economy class. Business class needs VP approval.
Allowance: Rs 2000/day domestic, $80/day international.
Hotel: Rs 5000/night domestic, $150 international.
Submit claims within 30 days. CFO approval above Rs 50000.""",
    metadata={"source": "Expense_Policy.pdf"}),
    Document(page_content="""PERFORMANCE REVIEWS
Reviews: June and December every year.
Self-assessment: 2 weeks before review date.
Rating: 1 (Needs Improvement) to 5 (Outstanding).
Promotion: Rating 4+ for two consecutive cycles.
Increments: Processed in January.""",
    metadata={"source": "Performance.pdf"}),
]

# ── Auto-build knowledge base on first load ──
if st.session_state.rag_chain is None:
    try:
        with st.spinner("⚙️ Setting up knowledge base, please wait..."):
            embeddings = load_embeddings()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(SAMPLE_DOCS)
            vs = FAISS.from_documents(chunks, embeddings)
            chain, retriever = build_rag_chain(vs)
            st.session_state.rag_chain = chain
            st.session_state.retriever = retriever
    except Exception as e:
        st.error(f"❌ Setup failed: {str(e)}")
        st.stop()
        embeddings = load_embeddings()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(SAMPLE_DOCS)
        vs = FAISS.from_documents(chunks, embeddings)
        chain, retriever = build_rag_chain(vs)
        st.session_state.rag_chain = chain
        st.session_state.retriever = retriever

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## 📂 Upload Documents")
    st.caption("Add your own PDF, TXT, or CSV files")
    uploaded_files = st.file_uploader(
        "Upload files", type=["pdf", "txt", "csv"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if st.button("➕ Add to Knowledge Base", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            new_docs = []
            with st.spinner("Loading files..."):
                for f in uploaded_files:
                    docs = load_document(f)
                    new_docs.extend(docs)
                    st.session_state.docs_loaded.append(f.name)
            if new_docs:
                with st.spinner("Updating knowledge base..."):
                    embeddings = load_embeddings()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    all_docs = SAMPLE_DOCS + new_docs
                    chunks = splitter.split_documents(all_docs)
                    vs = FAISS.from_documents(chunks, embeddings)
                    chain, retriever = build_rag_chain(vs)
                    st.session_state.rag_chain = chain
                    st.session_state.retriever = retriever
                st.success(f"✅ Added {len(new_docs)} new documents!")

    st.markdown("---")
    st.markdown("**📄 Built-in documents:**")
    for doc in SAMPLE_DOCS:
        st.caption(f"• {doc.metadata['source']}")
    if st.session_state.docs_loaded:
        st.markdown("**📄 Uploaded files:**")
        for name in st.session_state.docs_loaded:
            st.caption(f"• {name}")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    st.caption("Built with LangChain · FAISS · Groq · Streamlit")

# ── MAIN ──
st.markdown('<div class="main-header">🏢 Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your company policies, HR, IT, onboarding and more.</div>', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    qs = [
        "How many annual leave days do I get?",
        "What is the password policy?",
        "How do I claim travel expenses?",
        "When are performance reviews?",
        "What happens in week 1 of onboarding?",
        "What is the sick leave policy?",
    ]
    for i, q in enumerate(qs):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your company knowledge base..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_chain.invoke(prompt)
            docs = st.session_state.retriever.invoke(prompt)
            sources = list({d.metadata.get("source", "Unknown") for d in docs})
        st.markdown(answer)
        if sources:
            st.info("📚 Sources: " + ", ".join(sources))
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer + f"\n\n📚 Sources: {', '.join(sources)}"
        })
