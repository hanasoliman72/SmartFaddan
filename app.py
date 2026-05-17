import streamlit as st
import os
import sqlite3
import requests
from typing import Literal, Annotated, TypedDict
from dotenv import load_dotenv
# LangChain/LangGraph Imports
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

st.set_page_config(
    page_title="Smart Fadan | AI Farming",
    page_icon="🌱",
    layout="wide"
)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.75), rgba(255, 255, 255, 0.75)), 
                    url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] { display: none; }
    
    .hero-container {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(15px);
        border-radius: 40px;
        padding: 20px 10px;
        margin: 10px auto;
        max-width: 1000px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
    }
    
    h1 { font-family: 'serif'; color: #1b5e20; font-size: 4.2rem !important; letter-spacing: -2px; }

    /* --- THE CENTERED, WIDE, MORPHING BUTTON --- */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 900px !important;
        margin: 40px 0 !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1b5e20 0%, #43a047 100%) !important;
        color: white !important;
        
        /* Initial State: Sharp Top-Right and Bottom-Left */
        border-radius: 60px 0px 60px 0px !important;
        
        /* Dimensions: Wide and Sleek */
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        padding: 12px 100px !important; 
        width: 90% !important;
        max-width: 850px !important;
        height: auto !important;
        
        border: none !important;
        box-shadow: 0 8px 20px rgba(27, 94, 32, 0.2) !important;
        transition: all 0.4s ease-in-out !important;
        animation: none !important; 
    }

    /* HOVER STATE: Corner Flip */
    .stButton>button:hover {
        /* Morphs to: Sharp Top-Left and Bottom-Right */
        border-radius: 0px 60px 0px 60px !important;
        background: linear-gradient(135deg, #2e7d32 0%, #689f38 100%) !important;
        box-shadow: 0 12px 30px rgba(27, 94, 32, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    .stButton>button:active {
        transform: scale(0.98) !important;
    }

    .garden-plot {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 30px;
        padding: 40px;
        border-bottom: 8px solid #8bc34a;
        box-shadow: 0 15px 35px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    crop: str
    location: str
    rag_results: str
    weather_results: str
    final_answer: str
    needs_weather: bool

@st.cache_resource
def init_agent():
    llm = ChatOllama(model="qwen:1.8b", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    vectordb = PineconeVectorStore.from_existing_index(index_name="crop-index", embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    def router_node(state: AgentState):
        msg = state["messages"][-1].content.lower()
        needs_weather = any(word in msg for word in ["weather", "temperature", "rain", "climate"])
        return {"needs_weather": needs_weather}

    def rag_node(state: AgentState):
        query = f"{state.get('crop', '')} {state['messages'][-1].content}"
        docs = retriever.invoke(query)
        content = "\n\n".join(d.page_content for d in docs)
        return {"rag_results": content}

    def weather_node(state: AgentState):
        if not state.get("needs_weather"): return {"weather_results": "N/A"}
        loc = state.get("location", "Cairo")
        try:
            r = requests.get(f"https://wttr.in/{loc}?format=j1").json()
            curr = r["current_condition"][0]
            return {"weather_results": f"Temp: {curr['temp_C']}°C, Humidity: {curr['humidity']}%"}
        except: return {"weather_results": "Weather data unavailable"}

    def diagnosis_node(state: AgentState):
        prompt = f"As an agronomist, diagnose {state['crop']} in {state['location']}. Symptoms: {state['messages'][-1].content}. Context: {state['rag_results']}. Weather: {state['weather_results']}"
        ans = llm.invoke(prompt).content
        return {"final_answer": ans}

    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("diagnosis", diagnosis_node)
    
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "rag")
    workflow.add_edge("rag", "weather")
    workflow.add_edge("weather", "diagnosis")
    workflow.add_edge("diagnosis", END)
    
    conn = sqlite3.connect("farming_memory.sqlite3", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)

agent_app = init_agent()

st.markdown("""
    <div class="hero-container">
        <div style="font-size: 6rem;">🚜</div>
        <h1>Smart Fadan</h1>
        <p style="color: #2e7d32; font-weight: 500;">Intelligent Agriculture for the Modern Farmer</p>
    </div>
""", unsafe_allow_html=True)

_, col_mid, _ = st.columns([1, 5, 1])

with col_mid:
    st.subheader("🔍 Field Assessment")
    
    c1, c2 = st.columns(2)
    with c1:
        crop_input = st.text_input("Which crop are you growing?", placeholder="e.g. Wheat")
    with c2:
        gov_input = st.selectbox("Governorate", ["Assiut", "Beheira", "Dakahlia", "Giza", "Minya", "Sharqia"])
    
    symptoms_input = st.text_area("Describe the symptoms", 
                               placeholder="e.g. Yellow spots on lower leaves, wilting stems...",
                               height=120)
    
    if st.button("DIAGNOSE "):
        if crop_input and symptoms_input:
            with st.spinner('🌱 Analyzing your field data...'):
                config = {"configurable": {"thread_id": "streamlit_user"}}
                user_msg = f"In {gov_input}, my {crop_input} shows: {symptoms_input}"
                inputs = {
                    "messages": [HumanMessage(content=user_msg)],
                    "crop": crop_input,
                    "location": gov_input
                }
                output = agent_app.invoke(inputs, config)
                
                st.markdown("---")
                st.success("### 📋 AI Diagnosis & Recommendations")
                st.write(output["final_answer"])
        else:
            st.warning("⚠️ Please provide both the crop name and a description of the symptoms.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #4e342e; padding: 40px; font-size: 0.9rem; opacity: 0.7;">
        Smart Fadan AI • Cultivating Sustainable Futures since 2026
    </div>
""", unsafe_allow_html=True)