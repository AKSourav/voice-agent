import google.generativeai as genai

class LLM:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

        self.system_prompt = (
            "You are a concise AI assistant. "
            "Always respond clearly and helpfully, "
            "but never use more than 3 sentences in your reply."
        )

    def ask(self, text):
        res = self.model.generate_content(
            f"{self.system_prompt}\n\nUser: {text}",
            stream=True
        )
        response_text = ""
        for chunk in res:
            if chunk.text:
                response_text += chunk.text
        return response_text
