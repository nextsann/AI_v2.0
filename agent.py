import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import create_agent
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
        Creates a v1 Agent. 
        In LangChain v1.x, create_agent returns a CompiledGraph (Runnable).
        """
        # The v1 API simplifies everything into this single constructor
        return create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=f"You are {name}. {system_prompt}"
        )

    def create_agent_as_tool(self, name: str, system_prompt: str, tools: list, description: str):
        """
        Wraps a sub-agent as a tool.
        """
        agent_runner = self.create_agent(name, system_prompt, tools)
        
        def run_agent(query: str):
            # v1 Agents (Graph-based) expect a dict with "messages"
            inputs = {"messages": [HumanMessage(content=query)]}
            
            try:
                result = agent_runner.invoke(inputs)
                # v1 Response Extraction:
                # The result is the final state, so we get the last message's content
                return result["messages"][-1].content
            except Exception as e:
                return f"Error executing {name}: {e}"

        return Tool(
            name=name,
            func=run_agent,
            description=description
        )