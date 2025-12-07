import langchain
print(f"Your LangChain Version: {langchain.__version__}")

try:
    from langchain.agents import create_tool_calling_agent
    print("✅ SUCCESS: The function exists!")
except ImportError:
    print("❌ FAIL: The function is missing in this version.")