import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import pytz

# Import modules
from agent_factory import AgentFactory
from tools_library import get_search_tool, calendar_tools, email_tools

st.set_page_config(page_title="Mimi - Enterprise", page_icon="💃")
st.title("Mimi (v1.1.2 Architecture)")

if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ Groq API Key missing.")
    st.stop()
    
# --- INITIALIZE FACTORY ---
factory = AgentFactory()

# --- CREATE SPECIALIST AGENTS ---
# 1. Research
research_tool = get_search_tool()
if research_tool:
    research_agent = factory.create_agent_as_tool(
        name="Research_Specialist",
        system_prompt="Search Tavily and summarize findings.",
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
    system_prompt="Read unread emails or send new emails. Be concise.",
    tools=email_tools,
    description="Read or send emails."
)

# --- ROOT AGENT ---
specialists = [t for t in [research_agent, calendar_agent, email_agent] if t is not None]

current_time = datetime.now(pytz.timezone('Europe/London')).strftime("%A, %I:%M %p")

mimi = factory.create_agent(
    name="Mimi_Root",
    system_prompt=f"""
    Current Time: {current_time}
    You are Mimi, the Chief of Staff. 
    Delegate tasks to your specialists.
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
        with st.status("Working...", expanded=True) as status:
            # v1 Agents accept dictionary with 'messages' key
            response_state = mimi.invoke({"messages": st.session_state.messages})
            status.update(label="Done", state="complete", expanded=False)
            
        # CORRECT v1 EXTRACTION: Get text from the last message in the state
        final_answer = response_state["messages"][-1].content
        st.markdown(final_answer)
    
    st.session_state.messages.append(AIMessage(content=final_answer))