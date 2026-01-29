import os
from typing import TypedDict
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.messages import RemoveMessage


# =========================
# 1️⃣ DEFINE STRUCTURED OUTPUT
# =========================
class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    note: str


# =========================
# 2️⃣ CUSTOM STATE (SHORT TERM MEMORY)
# =========================
class CustomState(AgentState):
    user_name: str | None = None


class CustomContext(TypedDict):
    user_id: str


# =========================
# 3️⃣ TOOLS
# =========================
@tool
def get_user_name(runtime: ToolRuntime[CustomContext, CustomState]) -> str:
    """Get user name from database (fake)."""
    user_id = runtime.context["user_id"]
    name = "Royal" if user_id == "1" else "Guest"
    return f"User name is {name}"


@tool
def get_weather(city: str) -> WeatherResponse:
    """Get fake weather data for a city."""
    return WeatherResponse(
        temperature=30.0,
        condition="sunny",
        note=f"It's always bright in {city} 😎"
    )


# =========================
# 4️⃣ MEMORY TRIMMING MIDDLEWARE
# =========================
@before_model
def trim_messages(state: AgentState, runtime):
    messages = state["messages"]
    if len(messages) <= 6:
        return None

    first_msg = messages[0]
    recent = messages[-4:]
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first_msg,
            *recent
        ]
    }


# =========================
# 5️⃣ LOAD GEMINI MODEL
# =========================
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)


# =========================
# 6️⃣ CREATE AGENT WITH MEMORY
# =========================
agent = create_agent(
    model=model,
    tools=[get_user_name, get_weather],
    state_schema=CustomState,
    context_schema=CustomContext,
    middleware=[trim_messages],
    response_format=ToolStrategy(WeatherResponse),
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "thread-1"}}


# =========================
# 7️⃣ RUN CONVERSATION
# =========================
print("---- Conversation 1 ----")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, what's the weather in Hanoi?"}]},
    config=config,
    context={"user_id": "1"}
)

print(response["structured_response"])


print("\n---- Conversation 2 (memory continues) ----")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "And what about tomorrow?"}]},
    config=config,
    context={"user_id": "1"}
)

print(response["structured_response"])
