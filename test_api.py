import google.generativeai as genai

# Truyền thẳng API key vào đây
genai.configure(api_key="AIzaSyAIG8yt4lwsWZ800NiA1naKWMKIcfd9qjs")

# Tạo model Gemini
model = genai.GenerativeModel("gemini-2.5-flash")

# Gửi prompt
response = model.generate_content("Viết một đoạn thơ 4 câu về trí tuệ nhân tạo.")

# In kết quả
print(response.text)
