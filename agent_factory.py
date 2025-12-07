import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

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
        Creates a LangGraph ReAct Agent (The modern standard for v1.1.2).
        """
        # In LangGraph, we pass the system prompt as a 'state_modifier'
        # This prepends the system instructions to the message history.
        return create_react_agent(
            model=self.llm,
            tools=tools,
            state_modifier=f"You are {name}. {system_prompt}"
        )

    def create_agent_as_tool(self, name: str, system_prompt: str, tools: list, description: str):
        """
        Wraps a sub-agent as a tool for the root agent.
        """
        # Create the graph for the sub-agent
        agent_graph = self.create_agent(name, system_prompt, tools)
        
        def run_agent(query: str):
            # LangGraph inputs are just a dictionary of messages
            inputs = {"messages": [HumanMessage(content=query)]}
            
            # invoke() returns a dictionary of the final state (all messages)
            result = agent_graph.invoke(inputs)
            
            # The last message in the state is the AI's final answer
            return result["messages"][-1].content

        return Tool(
            name=name,
            func=run_agent,
            description=description
        )