from google import generativeai as genai


class LLM:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

    def ask(self, text):
        res = self.model.generate_content(text)
        return res.text
