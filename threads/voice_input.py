from vosk import Model, KaldiRecognizer
import pyaudio
from PyQt6.QtCore import QThread, pyqtSignal
import json

class VoskVoiceInputThread(QThread):
    """使用Vosk的语音输入线程"""
    text_available = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            # 初始化模型（需要指定模型路径）
            model = Model("resources/vosk-model-small-cn-0.22")
            recognizer = KaldiRecognizer(model, 16000)
            
            # 初始化音频流
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1,
                           rate=16000, input=True, frames_per_buffer=8192)
            stream.start_stream()
            
            self.text_available.emit("请开始说话...")
            
            while True:
                data = stream.read(4096)
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    text = json.loads(result)["text"]
                    if text:
                        self.text_available.emit(text)
                        break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            self.error_occurred.emit(f"语音识别错误: {str(e)}")