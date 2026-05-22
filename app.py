import os
import sqlite3
import requests
from typing import Literal
from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv
load_dotenv()

# LangChain core
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# LangChain community
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ── LLM & Embeddings ──────────────────────────────────────────────────────────

llm = ChatOllama(model="qwen2.5:7b", temperature=0)  # 0 => answers deterministic and consistent 

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── RAG Setup ─────────────────────────────────────────────────────────────────

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "crop-index"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def build_vectorstore():
    loader = PyPDFDirectoryLoader("pdfs/")
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages from PDFs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} chunks")

    vectordb = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=INDEX_NAME
    )
    print(f"✅ Vector store built with {len(chunks)} chunks")
    return vectordb

def load_vectorstore():
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

index = pc.Index(INDEX_NAME)
if index.describe_index_stats()["total_vector_count"] == 0:
    vectordb = build_vectorstore()
else:
    vectordb = load_vectorstore()
    print("✅ Vector store loaded from Pinecone")

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
print("✅ Retriever ready")

# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    crop: str
    location: str
    rag_results: str
    weather_results: str
    web_results: str
    final_answer: str
    needs_rag: bool
    needs_weather: bool
    needs_web: bool
    answer_is_sufficient: bool

# ── Graph Nodes ───────────────────────────────────────────────────────────────

def extract_info(state: AgentState) -> dict:
    latest_message = state["messages"][-1].content

    memory_context = ""
    if state.get("crop") or state.get("location"):
        memory_context = (
            f"Previously known context — Crop: {state.get('crop', 'unknown')}, "
            f"Location: {state.get('location', 'unknown')}. "
            "Use these if the user doesn't mention them again."
        )

    extract_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an agricultural assistant. Extract the crop name and Egyptian location "
            "from the user's message. " + memory_context + "\n"
            "Reply in this exact format:\n"
            "CROP: <crop name or 'unknown'>\n"
            "LOCATION: <location or 'unknown'>"
        )),
        ("human", "{message}")
    ])

    extract_chain = extract_prompt | llm | StrOutputParser()
    result = extract_chain.invoke({"message": latest_message})

    crop = state.get("crop", "unknown")
    location = state.get("location", "unknown")

    for line in result.strip().split("\n"):
        if line.startswith("CROP:"):
            extracted = line.replace("CROP:", "").strip()
            if extracted and extracted.lower() != "unknown":
                crop = extracted
        elif line.startswith("LOCATION:"):
            extracted = line.replace("LOCATION:", "").strip()
            if extracted and extracted.lower() != "unknown":
                location = extracted

    print(f"  [extract_info] crop={crop}, location={location}")
    return {"crop": crop, "location": location}


def router_node(state: AgentState) -> dict:
    latest_message = state["messages"][-1].content.lower()

    router_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a routing assistant for an agriculture support system. "
            "Decide which tools are needed to answer the user's message. "
            "Reply with ONLY a comma-separated list of needed tools from: rag, weather, web.\n"
            "Rules:\n"
            "- rag: if the question is about crop diseases, pests, treatment, or agricultural knowledge\n"
            "- weather: if the question mentions weather, temperature, humidity, or farming conditions\n"
            "- web: if the question is about current pest alerts, recent news, or market prices\n"
            "Example: rag,weather  or  rag  or  rag,weather,web"
        )),
        ("human", "{message}")
    ])

    routing_chain = router_prompt | llm | StrOutputParser()
    decision = routing_chain.invoke({"message": latest_message})

    tools_needed = [t.strip() for t in decision.lower().split(",")]

    needs_rag = "rag" in tools_needed
    needs_weather = "weather" in tools_needed
    needs_web = "web" in tools_needed

    if state.get("crop", "unknown") == "unknown":
        needs_rag = True

    print(f"  [router] needs_rag={needs_rag}, needs_weather={needs_weather}, needs_web={needs_web}")
    return {
        "needs_rag": needs_rag,
        "needs_weather": needs_weather,
        "needs_web": needs_web,
        "rag_results": "",
        "weather_results": "",
        "web_results": "",
        "answer_is_sufficient": False
    }


def rag_node(state: AgentState) -> dict:
    if not state.get("needs_rag", False):
        print("  [rag_node] skipped (not needed)")
        return {"rag_results": "Not needed for this query."}

    query = f"{state.get('crop', '')} {state['messages'][-1].content}"

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = ChatPromptTemplate.from_template(
        """You are an agricultural expert. Answer ONLY using the provided context. If the context doesn't contain enough information, say 'I don't have enough data on this' — do NOT use outside knowledge.
Context from agricultural documents:
{context}

Question: {question}

Provide relevant disease/pest information, symptoms, and treatments:"""
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(query)
    print(f"  [rag_node] retrieved info ({len(result)} chars)")
    return {"rag_results": result}


def weather_node(state: AgentState) -> dict:
    if not state.get("needs_weather", False):
        print("  [weather_node] skipped (not needed)")
        return {"weather_results": "Not needed for this query."}

    location = state.get("location", "unknown")
    if location == "unknown":
        location = "Cairo"
    location = location.replace(" ", "+")

    try:
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        weather_summary = (
            f"Location: {location}\n"
            f"Temperature: {current['temp_C']}°C\n"
            f"Feels like: {current['FeelsLikeC']}°C\n"
            f"Humidity: {current['humidity']}%\n"
            f"Description: {current['weatherDesc'][0]['value']}\n"
            f"Wind speed: {current['windspeedKmph']} km/h"
        )
        print(f"  [weather_node] fetched weather for {location}")
        return {"weather_results": weather_summary}

    except requests.exceptions.ConnectionError:
        return {"weather_results": "Weather unavailable: No internet connection"}
    except requests.exceptions.Timeout:
        return {"weather_results": "Weather unavailable: Request timed out"}
    except Exception as e:
        return {"weather_results": f"Weather unavailable: {e}"}


search_tool = TavilySearchResults(max_results=3)

def web_search_node(state: AgentState) -> dict:
    if not state.get("needs_web", False):
        print("  [web_search_node] skipped (not needed)")
        return {"web_results": "Not needed for this query."}

    crop = state.get("crop", "crop")
    location = state.get("location", "Egypt")
    query = f"{crop} pest alert disease treatment Egypt {location} agriculture 2024"

    try:
        results = search_tool.invoke(query)
        formatted = "\n".join(
            f"- {r.get('title', 'No title')}: {r.get('content', '')[:200]}"
            for r in results
        )
        print(f"  [web_search_node] found {len(results)} results")
        return {"web_results": formatted}
    except Exception as e:
        return {"web_results": f"Web search unavailable: {e}"}


def diagnosis_node(state: AgentState) -> dict:
    synthesis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an expert Egyptian agricultural advisor. "
            "Based on the information provided, give a clear diagnosis and actionable recommendations. "
            "Be specific, practical, and consider Egyptian farming conditions. "
            "Respond in the same language the user used (Arabic or English)."
        )),
        ("human", (
            "Farmer's question: {user_question}\n\n"
            "Crop: {crop}\n"
            "Location: {location}\n\n"
            "--- Agricultural Knowledge (from documents) ---\n{rag_results}\n\n"
            "--- Current Weather Conditions ---\n{weather_results}\n\n"
            "--- Current Pest Alerts & Tips (from web) ---\n{web_results}\n\n"
            "Provide a diagnosis and step-by-step recommendations:"
        ))
    ])

    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    answer = synthesis_chain.invoke({
        "user_question": state["messages"][-1].content,
        "crop": state.get("crop", "unknown"),
        "location": state.get("location", "unknown"),
        "rag_results": state.get("rag_results", "No data."),
        "weather_results": state.get("weather_results", "No data."),
        "web_results": state.get("web_results", "No data."),
    })

    print(f"  [diagnosis_node] answer generated ({len(answer)} chars)")
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)]
    }


def quality_check_node(state: AgentState) -> dict:
    check_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a quality reviewer. Judge if this agricultural advice is specific and actionable. "
            "Reply with ONLY: SUFFICIENT or INSUFFICIENT"
        )),
        ("human", "Answer to review:\n{answer}")
    ])

    quality_chain = check_prompt | llm | StrOutputParser()
    verdict = quality_chain.invoke({"answer": state.get("final_answer", "")})

    is_sufficient = "SUFFICIENT" in verdict.upper()
    print(f"  [quality_check] verdict={verdict.strip()}, is_sufficient={is_sufficient}")

    if not is_sufficient and state.get("retry_count", 0) < 1:
        return {"answer_is_sufficient": False, "needs_web": True, "retry_count": state.get("retry_count", 0) + 1}
   


def ask_clarification(state: AgentState) -> dict:
    message = (
        "I need a bit more information to help you. "
        "Could you please tell me:\n"
        "1. What crop are you growing?\n"
        "2. What is your location in Egypt?"
    )
    print("  [ask_clarification] requesting more info from user")
    return {
        "final_answer": message,
        "messages": [AIMessage(content=message)]
    }

# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_extract(state: AgentState) -> Literal["router", "ask_clarification"]:
    has_crop = state.get("crop", "unknown") != "unknown"
    has_location = state.get("location", "unknown") != "unknown"

    if has_crop or has_location:
        return "router"
    else:
        return "ask_clarification"


def route_after_rag(state: AgentState) -> Literal["weather", "web", "diagnosis"]:
    if state.get("needs_weather", False):
        return "weather"
    elif state.get("needs_web", False):
        return "web"
    else:
        return "diagnosis"


def route_after_weather(state: AgentState) -> Literal["web", "diagnosis"]:
    if state.get("needs_web", False):
        return "web"
    else:
        return "diagnosis"


def route_after_quality(state: AgentState) -> Literal["diagnosis", END]:
    if state.get("answer_is_sufficient", True):
        return END
    else:
        return "web"

# ── Build Graph ───────────────────────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("extract", extract_info)
graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("weather", weather_node)
graph.add_node("web", web_search_node)
graph.add_node("diagnosis", diagnosis_node)
graph.add_node("quality_check", quality_check_node)
graph.add_node("ask_clarification", ask_clarification)

graph.add_edge(START, "extract")

graph.add_conditional_edges(
    "extract",
    route_after_extract,
    {"router": "router", "ask_clarification": "ask_clarification"}
)

graph.add_edge("router", "rag")

graph.add_conditional_edges(
    "rag",
    route_after_rag,
    {"weather": "weather", "web": "web", "diagnosis": "diagnosis"}
)

graph.add_conditional_edges(
    "weather",
    route_after_weather,
    {"web": "web", "diagnosis": "diagnosis"}
)

graph.add_edge("web", "diagnosis")
graph.add_edge("diagnosis", "quality_check")

graph.add_conditional_edges(
    "quality_check",
    route_after_quality,
    {END: END, "web": "web"}
)

graph.add_edge("ask_clarification", END)

# ── Compile with SQLite Memory ────────────────────────────────────────────────

DB_PATH = "farming_agent_memory.sqlite3"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

app = graph.compile(checkpointer=memory)
print(f"✅ Agent compiled with SQLite memory at: {DB_PATH}")

# ── Chat Helper ───────────────────────────────────────────────────────────────

def chat(user_message: str, thread_id: str = "farmer-1") -> str:
    """
    Send a message to the farming agent and get a response.
    The thread_id preserves crop/location memory across calls.
    """
    config = {"configurable": {"thread_id": thread_id}}

    current_state = app.get_state(config)
    existing_crop = current_state.values.get("crop", "unknown") if current_state.values else "unknown"
    existing_location = current_state.values.get("location", "unknown") if current_state.values else "unknown"

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "crop": existing_crop,
            "location": existing_location,
            "rag_results": "",
            "weather_results": "",
            "web_results": "",
            "final_answer": "",
            "needs_rag": False,
            "needs_weather": False,
            "needs_web": False,
            "answer_is_sufficient": False
        },
        config
    )

    return result["final_answer"]

# ── Streamlit UI ──────────────────────────────────────────────────────────────

import streamlit as st

st.set_page_config(page_title="Smart Fadan", page_icon="🌱")

# Background image
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0)), url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌱 Smart Fadan")
st.write("Simple AI Farming Assistant")

# Inputs
symptoms = st.text_area("Describe Symptoms")

# Button
if st.button("Diagnose"):
    if symptoms:
        st.success("Diagnosis Result")
        user_message = f"I have the following symptoms: {symptoms}"
        with st.spinner("Analyzing..."):
            result = chat(user_message)
        st.write(result)
    else:
        st.warning("Please enter symptoms")