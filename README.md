# SafeX Solutions FAQ Chatbot 🤖

An AI-powered FAQ assistant for [safexsolutions.com](https://safexsolutions.com), built using **Retrieval-Augmented Generation (RAG)**. Instead of relying on rigid keyword matching, the bot understands the *meaning* behind a user's question, retrieves the most relevant company FAQs using semantic embeddings, and generates a natural, grounded answer.

Built during my internship at SafeX Solutions (AI/ML, DevOps & Cloud track).

## ✨ Features

- Semantic FAQ search using sentence embeddings — understands paraphrased questions, not just exact keyword matches
- LLM-generated, conversational answers grounded strictly in retrieved FAQ context (no hallucinated info)
- Lightweight FastAPI backend with a clean, responsive chat widget frontend
- Fully containerized with Docker + Docker Compose
- CI pipeline (GitHub Actions) for automated testing and image builds

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                             USER BROWSER                            │
│                     (static/index.html chat widget)                 │
└───────────────────────────────┬───────────────────────────────────┘
                                 │ POST /chat  { message }
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server (app.py)                      │
│                     serves frontend + /chat endpoint                │
└───────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG Engine (src/rag_bot.py)                     │
│                                                                       │
│   ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐    │
│   │   1. EMBED     │   │  2. RETRIEVE    │   │   3. GENERATE     │    │
│   │  User query -> │──▶│  Cosine        │──▶│  LLM answers      │    │
│   │  vector via    │   │  similarity    │   │  using ONLY the   │    │
│   │  sentence-     │   │  search over   │   │  retrieved FAQ    │    │
│   │  transformers  │   │  FAQ vector    │   │  context          │    │
│   │                │   │  store         │   │  (Claude API)     │    │
│   └───────────────┘   └────────┬───────┘   └──────────────────┘    │
│                                 ▲                                    │
│                                 │  pre-computed once at startup      │
│                        ┌────────┴────────┐                          │
│                        │  FAQ Knowledge   │                          │
│                        │  Base            │                          │
│                        │  (data/faqs.json)│                          │
│                        └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

**Pipeline breakdown:**

| Stage | What happens | Tech used |
|---|---|---|
| **Indexing** (once, at startup) | Every FAQ entry is converted into a dense vector embedding capturing its semantic meaning | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Retrieval** (per user query) | The user's question is embedded the same way; cosine similarity ranks all FAQs by relevance; top-k are selected | NumPy vector similarity |
| **Generation** (per user query) | Retrieved FAQ context + the user's question are passed to an LLM, which writes a grounded, natural-language answer | Anthropic Claude API |
| **Serving** | REST API + static chat widget | FastAPI + vanilla JS/HTML/CSS |
| **Packaging** | Reproducible runtime environment | Docker + Docker Compose |
| **Automation** | Tests + image build on every push | GitHub Actions |

## 📁 Project Structure

```
faq_chatbot/
├── data/
│   └── faqs.json           # FAQ knowledge base
├── src/
│   └── rag_bot.py           # RAG pipeline: embed → retrieve → generate
├── static/
│   └── index.html            # Chat widget UI
├── app.py                    # FastAPI server
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci-cd.yml
```

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/safex-faq-chatbot.git
cd safex-faq-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser.

## 📸 Screenshots

<!-- Add your screenshots below. Example markdown syntax to use once you have images: -->
<!-- ![Chat widget](docs/screenshots/chat-widget.png) -->

**Chat landing view**

<img width="960" height="485" alt="chatBot1" src="https://github.com/user-attachments/assets/3d925e71-ce14-4d59-8ac0-77a321f5847a" />


## 🛠️ Tech Stack

`Python` · `FastAPI` · `sentence-transformers` · `Anthropic Claude API` · `Docker` · `GitHub Actions`

## 📄 License

Specify your license here (e.g. MIT), or note that this is an internal SafeX Solutions project.

---
