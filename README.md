# 🏢 Enterprise Knowledge Assistant (RAG)

An AI-powered assistant that answers questions from company documents using **RAG (Retrieval-Augmented Generation)**. Built with LangChain, FAISS, Groq, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-purple)

---

## 🚀 Live Demo
Deployed on Streamlit Cloud — open the app and start chatting instantly. No setup required!

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (llama-3.3-70b-versatile) |
| Vector Database | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Framework | LangChain |
| Language | Python 3.11 |

---

## ✨ Features

- 💬 Chat interface to ask questions in plain English
- 🧠 RAG pipeline — answers grounded in real documents
- 📚 5 built-in sample enterprise documents
- 📂 Upload your own PDF, TXT, or CSV files
- 🔍 Source citations with every answer
- ⚡ Auto-builds knowledge base on app load
- 🌐 Deployable on Streamlit Cloud for free

---

## 📂 Built-in Sample Documents

| Document | Content |
|---|---|
| HR_Policy.pdf | Leave management, annual/sick/maternity leave |
| IT_Security.pdf | Password policy, VPN, breach reporting |
| Onboarding.pdf | Week-by-week onboarding guide |
| Expense_Policy.pdf | Travel, hotel, daily allowance rules |
| Performance.pdf | Review cycles, ratings, promotion criteria |

---

## 🌐 Deploy on Streamlit Cloud

1. Fork this repo on GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app** → select your repo → main file: `app.py`
4. Go to **Settings → Secrets** and add:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
5. Click **Deploy** — live in ~2 minutes!

---

## ⚙️ Run Locally

1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/RAG-Enterprise-Model
cd RAG-Enterprise-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

4. Run the app:
```bash
streamlit run app.py
```

5. Open [http://localhost:8501](http://localhost:8501)

---

## 🔑 Get Free API Keys

| Service | Link | Cost |
|---|---|---|
| Groq API Key | [console.groq.com](https://console.groq.com) | Free |
| Streamlit Cloud | [streamlit.io/cloud](https://streamlit.io/cloud) | Free |

---

## 📁 Project Structure

```
RAG-Enterprise-Model/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore              ← Ignores secrets & cache
└── .streamlit/
    └── config.toml         ← App theme settings
```

---

## 🔄 How RAG Works

```
User asks a question
        ↓
Question converted to vector embedding
        ↓
FAISS searches for similar document chunks
        ↓
Top 3 relevant chunks retrieved
        ↓
Groq LLM generates answer from context
        ↓
Answer returned with source citations
```

---

## 📜 License
MIT License — free to use and modify.
