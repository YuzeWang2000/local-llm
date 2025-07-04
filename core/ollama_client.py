import requests
import json

class OllamaAPI:
    BASE_URL = "http://localhost:11434"
    
    def __init__(self, model="gemma3n"):
        self.model = model
        self.generate_context = []  # 对话上下文
        self.chat_context = []  # 对话上下文
        
    def generate_response(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "context": self.generate_context,
            "stream": False,
            # "options": {"temperature": 0.7, "num_ctx": 2048, "seed": 42}
            "options": {"temperature": 0.7, "num_ctx": 2048}
        }
        
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"Response data: {data}")  # 调试输出
            # 更新上下文
            self.generate_context = data.get("context", [])
            return data["response"]
            
        except Exception as e:
            return f"错误: {str(e)}"
    
    def chat_response(self, prompt):
        self.chat_context.append({"role": "user", "content": prompt})
        print(self.chat_context)  # 调试输出
        payload = {
            "model": self.model,
            "messages": self.chat_context,
            "stream": False,
            "options": {"temperature": 0.7, "num_ctx": 2048}
        }
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            # 更新上下文
            self.chat_context.append(data["message"])
            print(self.chat_context)  # 调试输出
            return data["message"]["content"]
            
        except Exception as e:
            return f"错误: {str(e)}"
    def reset_context(self):
        self.generate_context = []
        self.chat_context = []
    
    def list_models(self):
        try:
            response = requests.get(f"{self.BASE_URL}/api/tags")
            return response.json().get("models", [])
        except:
            return []