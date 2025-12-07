import streamlit as st
from langchain_groq import ChatGroq
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
        Creates an agent using your simplified syntax.
        It builds the required PromptTemplate internally.
        """
        # 1. Create the template (Hidden from main logic)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are {name}. {system_prompt}"),
            MessagesPlaceholder(variable_name="messages"),
            ("placeholder", "{agent_scratchpad}"), 
        ])

        # 2. Create the Agent (Standard Tool Calling)
        agent = create_tool_calling_agent(
            model=self.llm,
            tools=tools,
            prompt=prompt
        )

        # 3. Return the Executor
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def create_agent_as_tool(self, name: str, system_prompt: str, tools: list, description: str):
        """
        Wraps a sub-agent as a tool for the root agent.
        """
        # Re-uses the clean create_agent method above
        agent_executor = self.create_agent(name, system_prompt, tools)
        
        def run_agent(query: str):
            try:
                # Wrap the string in the expected format
                response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
                return response.get("output", "No response.")
            except Exception as e:
                return f"Error: {e}"

        return Tool(
            name=name,
            func=run_agent,
            description=description
        )