# Setlu

Setlu is a conversational agent to let recruiters and hiring managers explore my professional profile through natural conversation. Instead of reading a static CV, you can just ask questions — and get answers tailored to what you actually care about.

**Live demo:** [maguettemb.github.io/Setlu](https://maguettemb.github.io/Setlu/)

---

## What it does

You pick a profile when you open the app — Recruiter, Technical Hiring Manager, or General — and the agent adapts its tone and focus accordingly. Ask about my research, my technical skills, my publications, whatever. It streams the answer back in real time.

Under the hood, it retrieves relevant chunks from my CV and project documents, reranks them using a second LLM call, and generates a grounded response. Conversations are stateful, so follow-up questions work naturally.

---

## How it's built

The retrieval combines FAISS (semantic search) and BM25 (keyword search) through LangChain's `EnsembleRetriever`. Retrieved chunks are scored and reranked in parallel by `gpt-4o-mini` before being passed to the main model. The backend is a FastAPI app deployed on Railway; the frontend is React + Vite on GitHub Pages.

```
app/
├── data/           # my CV, projects, publications (source documents)
├── vectorstore/    # FAISS index built by ingest.py
├── eval_reports/   # evaluation results
├── config.py
├── chatbot.py      # RAG chain, reranking, streaming
├── ingest.py       # document ingestion
├── evaluate.py     # evaluation pipeline
└── main.py         # FastAPI endpoints

setlu-frontend/
└── src/App.jsx     # React frontend
```

---

## Running it locally

You'll need Python 3.10+, Node 18+, and an OpenAI API key.

```bash
git clone https://github.com/maguettemb/Setlu.git
cd Setlu

pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY

python -m app.ingest           # build the vector store
uvicorn app.app:app --reload  # start the API on :8000
```

```bash
cd setlu-frontend
npm install
npm run dev
```

---

## API

| | |
|---|---|
| `POST /chat` | standard response |
| `POST /chat/stream` | streaming via SSE |
| `GET /profiles` | list available profiles |
| `GET /health` | health check |

---

## Evaluation

```bash
python -m app.evaluate
```

This runs a set of test questions and scores each answer on latency, precision (keyword overlap), faithfulness, relevancy, and completeness. The last two are judged by `gpt-4o-mini`. Reports land in `app/eval_reports/`.

Starting point before optimizations: ~7.7s average latency, 63% precision. Parallel reranking and hybrid retrieval brought both numbers down significantly.

---