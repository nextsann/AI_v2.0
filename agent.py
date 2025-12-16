import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import pytz
import rag_manager
import db
import uuid

# Import modules
from agent_factory import AgentFactory
from tools_library import get_search_tool, calendar_tools, email_tools, rag_tools

st.set_page_config(page_title="Mimi - Enterprise", page_icon="💃")
st.title("Mimi (Context-Aware Edition)")

if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ Groq API Key missing.")
    st.stop()

#SIDEBAR
with st.sidebar:


    with st.expander("🕵️ Debug: Brain Scan"):
        if st.button("Show all memories"):
            try:
                # Fetch the raw text from Supabase
                data = db.supabase.table("documents").select("content").limit(5).execute()
                if data.data:
                    for i, doc in enumerate(data.data):
                        st.text_area(f"Memory Chunk {i+1}", doc['content'], height=100)
                else:
                    st.warning("The database is empty! Did the upload fail?")
            except Exception as e:
                st.error(f"Debug Error: {e}")

    st.header("🧠 Knowledge Base")
    # PDF Uploader
    uploaded_file = st.file_uploader("Upload PDF (Internal Docs)", type=["pdf"])
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Ingesting document..."):
            result = rag_manager.ingest_pdf(uploaded_file)
            if "Success" in result:
                st.success(result)
            else:
                st.error(result)
    
    st.divider()
    
    st.header("🗄️ Chat History")
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # Load Past Sessions from Supabase
    try:
        sessions = db.get_all_sessions()
        for s in sessions:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"💬 {s['title']}", key=s['id'], use_container_width=True):
                    st.session_state.session_id = s['id']
                    # Load messages from DB -> Convert to LangChain objects
                    raw_msgs = db.get_messages(s['id'])
                    st.session_state.messages = [
                        HumanMessage(content=m['content']) if m['role'] == 'user' 
                        else AIMessage(content=m['content']) 
                        for m in raw_msgs
                    ]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{s['id']}"):
                    db.delete_session(s['id'])
                    if st.session_state.session_id == s['id']:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                    st.rerun()
    except Exception as e:
        st.warning(f"Could not load history: {e}")

# --- INITIALIZE FACTORY ---
factory = AgentFactory()

# --- CREATE SPECIALIST AGENTS ---
# 1. Research
research_tool = get_search_tool()
if research_tool:
    research_agent = factory.create_agent_as_tool(
        name="Research_Specialist",
        system_prompt="Search Tavily and summarize findings. Make sure you always send the information in reverse chronological order. Trust the query's specific details over your general knowledge.",
        tools=[research_tool],
        description="Search for news, facts, or web info."
    )
else:
    research_agent = None 

# 2. Calendar
calendar_agent = factory.create_agent_as_tool(
    name="Calendar_Specialist",
    system_prompt="Manage calendar events. Use ISO format.",
    tools=calendar_tools,
    description="Check schedule or create calendar events."
)

# 3. Email
email_agent = factory.create_agent_as_tool(
    name="Communication_Specialist",
    system_prompt="Read unread emails or send new emails. Be concise, but follow 100% the email body you were sent, do not change it!.",
    tools=email_tools,
    description="Read or send emails."
)

#RAG
knowledge_agent = factory.create_agent_as_tool(
    name="Knowledge_Specialist",
    system_prompt="""You are the keeper of the user's personal history and preferences.
    The database contains their travel logs, favourite foods, friends, and past projects.
    
    RULE: If the user asks a question about THEMSELVES (e.g., "What do I like?", "Where should I go?"), 
    you MUST query the database first. Do not assume you don't know.""",
    tools=rag_tools,
    description="The FIRST place to check for ANY question about the user's preferences, history, or files."
)

# --- ROOT AGENT ---
specialists = [t for t in [research_agent, calendar_agent, email_agent, knowledge_agent] if t is not None]

london_tz = pytz.timezone('Europe/London')
current_full_date = datetime.now(london_tz).strftime("%A, %B %d, %Y")
current_time = datetime.now(london_tz).strftime("%I:%M %p")

# We update the System Prompt to force "Query Expansion"
mimi = factory.create_agent(
    name="Mimi_Root",
    system_prompt=f"""
    CONTEXT:
    - Today is: {current_full_date}
    - Time is: {current_time}
    
    You are Mimi, the Chief of Staff.
    
    CRITICAL INSTRUCTION - QUERY REWRITING:
    The specialists (Research, Calendar, Email) DO NOT have access to the chat history.
    If the user asks a follow-up question like "When was that?" or "Who won?", you MUST rewrite the query to include the full context.
    
    Examples:
    - User: "When was that?" (after discussing Liverpool vs Leeds) -> Tool Input: "Date of Liverpool vs Leeds match December 2025"
    - User: "Send it to him" (after discussing Bob) -> Tool Input: "Send email to Bob..."
    
    Delegate tasks to your specialists:
    1. Research Specialist (News, Sports, Weather)
    2. Calendar Specialist (Schedule)
    3. Communication Specialist (Email)
    4. Knowledge Specialist -> First stop for user preferences, internal docs, and context.
    """,
    tools=specialists
)

# --- CHAT LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if isinstance(msg, (HumanMessage, AIMessage)):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

if user_input := st.chat_input("How can I help?"):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            response_state = mimi.invoke({"messages": st.session_state.messages})
            status.update(label="Done", state="complete", expanded=False)
            
        final_answer = response_state["messages"][-1].content
        st.markdown(final_answer)
    
    st.session_state.messages.append(AIMessage(content=final_answer))
