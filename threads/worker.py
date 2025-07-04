
from PyQt6.QtCore import QThread, pyqtSignal

class GenerateWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, ollama_api, prompt):
        super().__init__()
        self.api = ollama_api
        self.prompt = prompt        

    def run(self):
        try:
            response = self.api.generate_response(self.prompt)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))

class ChatWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, ollama_api, prompt):
        super().__init__()
        self.api = ollama_api
        self.prompt = prompt
        
    def run(self):
        try:
            response = self.api.chat_response(self.prompt)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))

