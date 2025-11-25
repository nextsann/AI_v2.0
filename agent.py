import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Mimi - LangChain", page_icon="💃")
st.title("Mimi Bebesita (LangChain Edition)")

# Check for API Key
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ API Key missing. Please add GEMINI_API_KEY to secrets.toml")
    st.stop()

# --- 2. INITIALIZE LANGCHAIN MODEL ---
# This is the wrapper that makes Gemini speak "LangChain"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=st.secrets["GEMINI_API_KEY"],
    temperature=0.7
)

# --- 3. SESSION STATE (Chat History) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. RENDER UI ---
# Draw existing messages
for msg in st.session_state.messages:
    # Streamlit uses "user" and "assistant" roles
    # LangChain uses "HumanMessage" and "AIMessage" objects
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- 5. HANDLE INPUT ---
if prompt := st.chat_input("What is on your mind?"):
    
    # A. Display User Message
    st.chat_message("user").markdown(prompt)
    
    # B. Add to History (as a LangChain Object)
    st.session_state.messages.append(HumanMessage(content=prompt))

    # C. Generate Response using LangChain
    with st.chat_message("assistant"):
        # stream() is a cool LangChain feature that types the text out live
        response = st.write_stream(llm.stream(st.session_state.messages))
    
    # D. Add AI Response to History
    st.session_state.messages.append(AIMessage(content=response))