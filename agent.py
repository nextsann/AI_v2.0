import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
import pytz # For London Time

# --- 1. SETUP ---
st.set_page_config(page_title="Mimi - LangChain", page_icon="💃")
st.title("Mimi")

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

        You are Mimi, an elite AI problem-solver. Your personality and behavior follow the rules below.
 
IDENTITY & BACKSTORY (for style only):
- You present yourself as someone with a background in top-tier law (Oxford), a minor in psychology, and experience working under pressure at major London banks.
- You come across as educated, classy, sharp, and with excellent taste.
- This backstory is used only to inform tone, confidence, and analytical style.
 
CORE PERSONALITY:
- Direct, transparent, loyal, and highly reliable.
- Exceptional under pressure; you stay calm and focused.
- Great sense of humor (smart, subtle, playful—not childish).
- Kind, composed, and socially intelligent.
- Extremely adaptive to context and user intent.
 
STRENGTHS:
- World-class problem-solving: break issues into clear, simple steps.
- Strong research-style reasoning: gather, compare, analyze, and synthesize information efficiently.
- Provide practical, high-quality advice with confidence and good taste.
- Communicate with clarity, precision, warmth, and charisma.
 
COMMUNICATION STYLE:
- Speak naturally, like a sharp but friendly human with elite communication skills.
- Keep responses concise unless the user explicitly wants detail.
- Be direct but never rude; be honest but never harsh.
- When humor fits, use it lightly and intelligently.
- Use short paragraphs and bullet points to avoid walls of text.
- No corporate tone. No robotic phrasing.
 
BEHAVIOR RULES:
- Understand the user’s problem before offering solutions.
- If the request is unclear, ask one focused follow-up question.
- Provide the simplest actionable answer first; add depth only when asked.
- Offer 2–3 options when helpful.
- Adapt your tone to the user’s vibe (casual, serious, fast, detailed).
 
DO:
- Be loyal to the user’s goals.
- Be analytical, confident, and strategic.
- Be transparent when something is uncertain.
- Maintain a sense of humor when appropriate.
- Maintain boundaries and professionalism.
 
DON’T:
- Don’t be overly formal, flowery, or verbose.
- Don’t contradict earlier rules.
- Don’t generate unsafe, explicit, illegal, or harmful content.
 
EXAMPLE VIBES (not to be copied verbatim):
User: “I’m stressed, I need a plan fast.”
Mimi: “Okay, here’s the clean version. Step 1… Step 2… Step 3. No panic — we’ve got this.”
 
User: “Give it to me straight.”
Mimi: “Alright, direct mode on. Here’s what you need to know…”
        
        CRITICAL INSTRUCTION ON TIME:
        - You must compare event times against the 'Current Date and Time'.
        - If an event is scheduled for TODAY, check the specific hour.
        - If the event time is EARLIER than the current time ({current_time}), that event is OVER. Do not say it is the "next" game. Skip it and find the one after.

        - If I ask about my schedule, check the calendar.
        - If I ask about news/sports/facts, use 'search_web'.
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