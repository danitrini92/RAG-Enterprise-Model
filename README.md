# 🏢 Enterprise Knowledge Assistant (RAG)

An AI-powered assistant that answers questions from company documents using RAG.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM:** Groq (llama-3.3-70b-versatile)
- **Vector DB:** FAISS
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Framework:** LangChain

## 🌐 Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Click **New app** → select repo → main file: `app.py`
4. Go to **Settings → Secrets** and add:
```
GROQ_API_KEY = "gsk_your_key_here"
```
5. Click **Deploy**!

## ⚙️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Add your key to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

## 📂 Supported Files
PDF, TXT, CSV
