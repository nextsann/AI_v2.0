import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

class AgentFactory:
    def __init__(self, model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.7):
        self.llm = ChatGroq(
            model=model_name, 
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=temperature,
            max_tokens=4096 
        )

    def create_agent(self, name: str, system_prompt: str, tools: list):
        """
        Creates a Tool Calling Agent (No LangGraph, uses AgentExecutor).
        """
        # 1. Define the Prompt
        # We must include placeholders for the tool agent to work
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are {name}. {system_prompt}"),
            MessagesPlaceholder(variable_name="messages"),
            ("placeholder", "{agent_scratchpad}"), 
        ])

        # 2. Construct the Agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        # 3. Create the Executor (The Runtime)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def create_agent_as_tool(self, name: str, system_prompt: str, tools: list, description: str):
        """
        Wraps a sub-agent as a tool for the root agent.
        """
        agent_executor = self.create_agent(name, system_prompt, tools)
        
        def run_agent(query: str):
            # The executor expects a dictionary with "messages"
            # It returns a dictionary with "output"
            try:
                # We wrap the string query in a HumanMessage
                response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
                return response.get("output", "No response generated.")
            except Exception as e:
                return f"Error running {name}: {e}"

        return Tool(
            name=name,
            func=run_agent,
            description=description
        )