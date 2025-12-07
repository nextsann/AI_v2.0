import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

class AgentFactory:
    def __init__(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct", 
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.7,
            max_tokens=4096 
        )

    def create_agent(self, name: str, system_prompt: str, tools: list):
        """
        Creates a Tool Calling Agent (Standard for Llama 3/4).
        """
        # 1. Define the Prompt Template internally
        # We must include "agent_scratchpad" for the tool calling agent to work
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are {name}. {system_prompt}"),
            MessagesPlaceholder(variable_name="messages"),
            ("placeholder", "{agent_scratchpad}"), 
        ])

        # 2. Construct the Agent
        agent = create_tool_calling_agent(
            model=self.llm,
            tools=tools,
            prompt=prompt
        )

        # 3. Create the Executor (The Runtime)
        # verbose=True helps you see the "thinking" process in the console
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def create_agent_as_tool(self, name: str, system_prompt: str, tools: list, description: str):
        """
        Wraps a sub-agent as a tool for the root agent.
        """
        agent_executor = self.create_agent(name, system_prompt, tools)
        
        def run_agent(query: str):
            # The executor expects a dictionary with "messages"
            try:
                # We wrap the string query in a HumanMessage
                response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
                
                # IMPORTANT: AgentExecutor returns 'output', NOT 'messages'
                return response.get("output", "No response generated.")
            except Exception as e:
                return f"Error running {name}: {e}"

        return Tool(
            name=name,
            func=run_agent,
            description=description
        )