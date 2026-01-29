import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

# ===== 1. Schema định dạng output =====
class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    note: str

# ===== 2. Load Gemini model =====
# Nhớ đã set biến môi trường GOOGLE_API_KEY trước đó
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

# Ép model trả về đúng schema
structured_llm = llm.with_structured_output(WeatherResponse)

# ===== 3. Gửi câu hỏi =====
result = structured_llm.invoke(
    "What's the weather in Hanoi today? "
    "Return realistic weather conditions."
)

# ===== 4. In kết quả =====
print("Structured object:", result)
print("Temperature:", result.temperature)
print("Condition:", result.condition)
print("Note:", result.note)
