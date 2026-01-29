import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    note: str

#Load Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

structured_llm = llm.with_structured_output(WeatherResponse)

result = structured_llm.invoke(
    "What's the weather in Hanoi today? "
    "Return realistic weather conditions."
)

#In kết quả
print("Structured object:", result) #Cấu trúc đầu ra dựa vào ví dụ weather
print("Temperature:", result.temperature)
print("Condition:", result.condition)
print("Note:", result.note)
