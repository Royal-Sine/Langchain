import requests
from markdownify import markdownify

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

#Tool
@tool
def fetch_documentation(url: str) -> str:
    """Fetch documentation page"""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return markdownify(response.text)

tools = [fetch_documentation]

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

#Prompt cho AI
SYSTEM_PROMPT = "You are an expert in LangGraph. Use tools when documentation is needed."

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Write a short example of a LangGraph agent that can look up stock prices."}
    ]
})

print(response["messages"][-1].content)
# Chưa chạy được!