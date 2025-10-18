import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
# Truyền thẳng API key vào đây
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Tạo model Gemini
model = genai.GenerativeModel("gemini-2.5-flash")

# Gửi prompt
response = model.generate_content("Viết một đoạn thơ 4 câu về trí tuệ nhân tạo.")

# In kết quả
print(response.text)
