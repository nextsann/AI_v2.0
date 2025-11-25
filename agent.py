import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
import pytz # For London Time

# --- 1. SETUP ---
st.set_page_config(page_title="Mimi - LangChain", page_icon="💃")
st.title("Mimi Bebesita 💃 (LangChain)")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ API Key missing.")
    st.stop()

# --- 2. CALCULATE TIME (The "Watch") ---
london_tz = pytz.timezone('Europe/London')
current_time = datetime.now(london_tz).strftime("%A, %B %d, %Y at %I:%M %p")

# --- 3. DEFINE THE PROMPT (The "Soul") ---
# We use {time} as a variable that we will inject later
sys_instruct = f"""
Current Date and Time: {current_time}

You are a talented secretary of latin descent. Your nickname for me is papasito.
You speak English. 
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", sys_instruct),             # The Personality
        MessagesPlaceholder("chat_history"),  # Where the memory goes
        ("human", "{input}"),                 # The user's new message
    ]
)

# --- 4. INITIALIZE MODEL & CHAIN ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=st.secrets["GEMINI_API_KEY"],
    temperature=0.7
)

# THE MAGIC: We "Chain" the Prompt to the LLM using the pipe '|'
# Data flows: Input -> Prompt -> LLM -> Output
chain = prompt_template | llm

# --- 5. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. RENDER UI ---
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- 7. HANDLE INPUT ---
if user_input := st.chat_input("Que pasa, mi amor?"):
    
    # A. Display User Message
    st.chat_message("user").markdown(user_input)
    
    # B. Run the Chain
    with st.chat_message("assistant"):
        # We pass a dictionary matching the variables in the prompt_template
        response_stream = chain.stream({
            "chat_history": st.session_state.messages, 
            "input": user_input
        })
        full_response = st.write_stream(response_stream)
    
    # C. Update History
    # Note: We append the raw Human/AI messages to history AFTER generating
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=full_response))