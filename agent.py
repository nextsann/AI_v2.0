import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import pytz

# Import our new modules
from agent_factory import AgentFactory
from tools_library import get_search_tool, calendar_tools, email_tools

# --- 1. SETUP UI ---
st.set_page_config(page_title="Mimi - Enterprise", page_icon="💃")
st.title("Mimi (Hierarchical Architecture)")

# Check Secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ Groq API Key missing. Please add it to secrets.toml")
    st.stop()
    
# --- 2. INITIALIZE FACTORY ---
# Llama 4.0 Maverick is set as default in the factory now
factory = AgentFactory()

# --- 3. CREATE SPECIALIST AGENTS ---

# A. Research Specialist
research_tool = get_search_tool()
if research_tool:
    research_agent_tool = factory.create_agent_as_tool(
        name="Research_Specialist",
        system_prompt="""You are a focused web researcher. 
        You search for accurate data using Tavily and summarize it concisely. 
        Always provide the source URL if possible.""",
        tools=[research_tool],
        description="Use this agent to search the internet for news, facts, sports scores, or current events."
    )
else:
    research_agent_tool = None 

# B. Calendar Specialist
calendar_agent_tool = factory.create_agent_as_tool(
    name="Calendar_Specialist",
    system_prompt="""You are a precise schedule manager.
    You can list and create calendar events.
    When creating events, ensure dates are formatted correctly (ISO format).
    Do not guess; if an ID is needed, list events first.""",
    tools=calendar_tools,
    description="Use this agent to manage the user's schedule, check appointments, or book meetings."
)

# C. Communication Specialist (NEW)
email_agent_tool = factory.create_agent_as_tool(
    name="Communication_Specialist",
    system_prompt="""You are a professional secretary handling emails.
    You can read unread emails and send new emails.
    When sending emails, be professional and concise.
    When reading emails, summarize the key points.""",
    tools=email_tools,
    description="Use this agent to read unread emails or send emails to people."
)

# --- 4. CREATE ROOT AGENT (MIMI) ---
# Filter out None in case Research tool failed
specialists = [research_agent_tool, calendar_agent_tool, email_agent_tool]
root_tools = [t for t in specialists if t is not None]

# Time logic for Mimi
london_tz = pytz.timezone('Europe/London')
current_time = datetime.now(london_tz).strftime("%A, %B %d, %Y at %I:%M %p")

mimi_executor = factory.create_agent(
    name="Mimi_Root",
    system_prompt=f"""
    Current Date: {current_time}
    
    You are Mimi, the Chief of Staff.
    You manage a team of specialists:
    1. Research Specialist (for web info)
    2. Calendar Specialist (for schedule info)
    3. Communication Specialist (for reading/sending emails)
    
    Delegate tasks to them. Do not try to do their jobs yourself.
    If the user asks something simple (chat), answer directly.
    If the user asks a complex question, decide which specialist to call.
    """,
    tools=root_tools
)

# --- 5. CHAT LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if isinstance(msg, (HumanMessage, AIMessage)):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

if user_input := st.chat_input("Que pasa?"):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            # We invoke the executor
            response = mimi_executor.invoke({"messages": st.session_state.messages})
            status.update(label="Complete", state="complete", expanded=False)
            
        # Extract output cleanly from AgentExecutor response
        final_answer = response.get("output", "I encountered an issue generating a response.")
        
        st.markdown(final_answer)
    
    st.session_state.messages.append(AIMessage(content=final_answer))