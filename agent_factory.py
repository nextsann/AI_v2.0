from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

class AgentFactory:
    def __init__(self, model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.7):
        self.llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct", 
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.7,
        max_tokens=4096 # Safety limit
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