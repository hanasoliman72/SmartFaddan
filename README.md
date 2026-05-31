# Smart Fadan — AI Farming Assistant

An agentic AI system that helps Egyptian farmers diagnose crop diseases 
and get actionable recommendations using RAG, real-time weather, and web search.

## How it works
The agent routes each query through a LangGraph pipeline:
- **RAG** — searches local agricultural PDFs via Pinecone
- **Weather** — fetches real-time conditions from wttr.in
- **Web Search** — finds current pest alerts via Tavily
- **Self-correction** — quality check node retries with web search if answer is insufficient
- **Memory** — remembers crop and location across conversation turns via SQLite

## Tech Stack
Python, LangGraph, LangChain, Ollama (Qwen2.5:7b), Pinecone, HuggingFace Embeddings, Tavily, Streamlit, SQLite

## Setup
```bash
pip install langchain langchain-community langgraph langgraph-checkpoint-sqlite
pip install tavily-python langchain-ollama pinecone langchain-pinecone
ollama pull qwen2.5:7b
```

Add a `.env` file:

PINECONE_API_KEY=your_key

TAVILY_API_KEY=your_key

Then run:
```bash
streamlit run app.py
```
