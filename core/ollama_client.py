import requests
import json
import ollama
from ollama import chat
from ollama import Client

class OllamaAPI:
    BASE_URL = "http://localhost:11434"
    
    def __init__(self, model="gemma3n"):
        self.model = model
        self.generate_context = []  # 对话上下文
        self.chat_context = []  # 对话上下文
        
    def generate_response(self, prompt):
 
        try:
            client = Client(
            host=self.BASE_URL,
            headers={'x-some-header': 'some-value'}
            )
            response = client.generate(model=self.model,
                                       prompt=prompt,
                                       context=self.generate_context,
                                       stream=False,
                                       options={"temperature": 0.7, "num_ctx": 2048})

            self.generate_context = response.context  # 更新上下文
            return response.response
            
        except Exception as e:
            return f"错误: {str(e)}"
    
    def chat_response(self, prompt):
        self.chat_context.append({"role": "user", "content": prompt})
        print(self.chat_context)  # 调试输出
        try:
            client = Client(
            host=self.BASE_URL,
            headers={'x-some-header': 'some-value'}
            )
            response = client.chat(model=self.model,
                                       messages=self.chat_context,
                                       stream=False,
                                       options={"temperature": 0.7, "num_ctx": 2048})

            print(f"Response: {response}")  # 调试输出
            print(f"Response messages: {response.message}")  # 调试输出
        
            # 更新上下文
            self.chat_context.append({"role": response.message.role, "content": response.message.content})
            print(self.chat_context)  # 调试输出
            return response.message.content
            
        except Exception as e:
            return f"错误: {str(e)}"
        
    def reset_context(self):
        self.generate_context = []
        self.chat_context = []
    
    def get_model_list(self):
        
        return ollama.list()