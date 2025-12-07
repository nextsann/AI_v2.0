from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

class AgentFactory:
    def __init__(self, model_name="gemini-1.5-flash", temperature=0.7):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )

    def create_agent(self, name: str, system_prompt: str, tools: list):
        """
        Creates a v1.x Agent (Replaces the old AgentExecutor pattern).
        """
        # In LangChain v1, create_agent handles the graph, tools, and execution loop automatically.
        agent_runner = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=f"You are the {name}. {system_prompt}",
        )
        
        return agent_runner

    def create_agent_as_tool(self, name: str, system_prompt: str, tools: list, description: str):
        """
        Wraps a sub-agent as a tool for the root agent.
        """
        agent_runner = self.create_agent(name, system_prompt, tools)
        
        def run_agent(query: str):
            # v1 Agents expect a dictionary with "messages"
            response = agent_runner.invoke({"messages": [HumanMessage(content=query)]})
            
            # The response format in v1 might differ slightly, usually returning the last message
            # We extract the text content safely
            if isinstance(response, dict) and "messages" in response:
                return response["messages"][-1].content
            return str(response)

        return Tool(
            name=name,
            func=run_agent,
            description=description
        )