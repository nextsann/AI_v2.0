from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

class AgentFactory:
    def __init__(self, model_name="gemini-2.5-flash", temperature=0.7):
        # We initialize the generic LLM configuration once
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )

    def create_agent(self, name: str, system_prompt: str, tools: list):
        """
        Creates a specific sub-agent executor (e.g., 'Calendar Agent').
        """
        # 1. Bind tools to this specific agent's brain
        llm_with_tools = self.llm.bind_tools(tools)

        # 2. Define the agent's specific instructions
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are the {name}. {system_prompt}"),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # 3. Create the Agent Runtime
        agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
        
        # 4. Wrap it in an Executor
        executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, # Set to True to see the sub-agent thinking in the logs
            handle_parsing_errors=True
        )
        
        return executor

    def create_agent_as_tool(self, name: str, system_prompt: str, tools: list, description: str):
        """
        Creates an agent and immediately wraps it as a Tool for the Root Agent.
        """
        executor = self.create_agent(name, system_prompt, tools)
        
        def run_agent(query: str):
            # The sub-agent receives the query as a new "human" message
            response = executor.invoke({"messages": [HumanMessage(content=query)]})
            return response["output"]

        return Tool(
            name=name,
            func=run_agent,
            description=description
        )