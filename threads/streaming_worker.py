from PyQt6.QtCore import QThread, pyqtSignal
class StreamingWorker(QThread):
    # 自定义信号
    partial_response = pyqtSignal(str)  # 部分响应信号
    finished = pyqtSignal()             # 处理完成信号
    error = pyqtSignal(str)              # 错误信号
    
    def __init__(self, api, prompt, mode, parent=None):
        super().__init__(parent)
        self.api = api
        self.prompt = prompt
        self.mode = mode
        self.cancel_requested = False
    
    def run(self):
        try:
            # 使用流式API生成响应
            if self.mode == "retrieval":
                response_stream = self.api.stream_rag_response(self.prompt)
            elif self.mode == "chat":
                response_stream = self.api.stream_chat_response(self.prompt)
            elif self.mode == "generate":
                response_stream = self.api.stream_generate_response(self.prompt)
            
            # 处理流式响应

            for chunk in response_stream:
                # print(f"接收到数据块: {chunk}")
                if self.cancel_requested:
                    self.partial_response.emit("已取消")
                    return
                
                if chunk and chunk.strip():  # 确保有有效内容
                    self.partial_response.emit(chunk)
            
            self.finished.emit()
        
        except Exception as e:
            print(f"处理错误: {str(e)}")
            self.error.emit(f"处理错误: {str(e)}")
    
    def cancel(self):
        self.cancel_requested = True