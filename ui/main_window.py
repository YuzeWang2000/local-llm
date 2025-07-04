from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, 
    QTextBrowser, QTextEdit, QPushButton,
    QHBoxLayout, QLabel, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt
from threads.worker import GenerateWorker, ChatWorker
from threads.voice_input import VoskVoiceInputThread
from core.ollama_client import OllamaAPI
import markdown
import pyttsx3  # 添加语音合成库
import re
class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama Chat")
        self.setGeometry(100, 100, 800, 600)
        self.current_response = ""  # 用于存储当前响应内容
        # 初始化语音引擎
        self.speaker = pyttsx3.init()
        self.speaker.setProperty('rate', 150)  # 设置语速
        self.speaker.setProperty('volume', 0.9)  # 设置音量
        # 初始化API客户端
        self.api = OllamaAPI()
        
        # 创建UI
        self._create_ui()
        
        # 加载模型列表
        self._load_models()

    
    def _create_ui(self):
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部控制栏
        control_layout = QHBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        
        self.clear_btn = QPushButton("清除历史")
        self.clear_btn.clicked.connect(self._clear_context)
        
        control_layout.addWidget(QLabel("选择模型:"))
        control_layout.addWidget(self.model_combo, 3)
        control_layout.addWidget(self.clear_btn)
        
        # 在控制栏添加语音按钮
        self.voice_btn = QPushButton("朗读")
        self.voice_btn.clicked.connect(self._speak_output)
        control_layout.addWidget(self.voice_btn)

        # 聊天显示区域
        self.output_area = QTextBrowser()
        self.output_area.setOpenExternalLinks(True)
        
        # 输入区域
        input_layout = QHBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("输入消息...")
        self.input_box.setMinimumHeight(100)
        
        # 语音输入 
        self.voice_input_btn = QPushButton("语音输入")
        self.voice_input_btn.clicked.connect(self._start_voice_input)
        
        # 在控制栏中添加按钮
        control_layout.addWidget(self.voice_input_btn)
        
        # 初始化语音输入线程
        self.voice_thread = None
        # 模式切换和显示
        mode_layout = QVBoxLayout()
        self.mode_label = QLabel("当前模式:")
        self.is_chat_mode = True  # 默认聊天模式
        self.mode_display = QLabel("聊天模式")

        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self._send_chat_message)

        # self.mode_display.setStyleSheet("font-weight: bold; color: #4ec9b0;")
        self.change_mode_btn = QPushButton("切换模式")
        self.change_mode_btn.clicked.connect(self._toggle_mode)

        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_display)
        mode_layout.addWidget(self.change_mode_btn)
        mode_layout.addWidget(self.send_btn)

        input_layout.addWidget(self.input_box, 8)
        input_layout.addLayout(mode_layout, 2)
        
        # 添加到主布局
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.output_area, 6)
        main_layout.addLayout(input_layout, 1)
    
    def _load_models(self):
        models = self.api.list_models()
        for m in models:
            print(f"加载模型: {m['name']}")  # 调试输出
        self.model_combo.addItems([m["name"] for m in models])
        # self.model_combo.setCurrentText("gemma3n:latest")
    
    def _speak_output(self):
        """朗读输出区域的内容"""
        try:
            # 获取输出区域的纯文本内容（去除HTML标签）
            plain_text = self.current_response
            print(f"朗读内容: {plain_text}")  # 调试输出
            plain_text = re.sub(r'<[^>]+>', '', plain_text)  # 移除所有HTML标签
            print(f"朗读内容: {plain_text}")  # 调试输出

            plain_text = re.sub(r'[^\w\s,.!?！？，。]', '', plain_text)
            print(f"朗读内容: {plain_text}")  # 调试输出

            plain_text = re.sub(r'\s+', ' ', plain_text).strip()  # 合并多余空格

            print(f"朗读内容: {plain_text}")  # 调试输出
            # plain_text = self.output_area.toPlainText()
            if not plain_text.strip():
                self.statusBar().showMessage("没有内容可朗读", 2000)
                return
                
            # 朗读内容
            self.speaker.say(plain_text)
            self.speaker.runAndWait()
            self.statusBar().showMessage("朗读完成", 2000)
            
        except Exception as e:
            self.statusBar().showMessage(f"朗读出错: {str(e)}", 3000)

    def _start_voice_input(self):
        """开始语音输入"""
        if self.voice_thread and self.voice_thread.isRunning():
            QMessageBox.information(self, "提示", "正在录音中，请稍候")
            return
            
        self.voice_thread = VoskVoiceInputThread()
        self.voice_thread.text_available.connect(self._handle_voice_input)
        self.voice_thread.error_occurred.connect(self._handle_voice_error)
        self.voice_thread.start()
        
        # 禁用按钮防止重复点击
        self.voice_input_btn.setEnabled(False)
        self.voice_thread.finished.connect(
            lambda: self.voice_input_btn.setEnabled(True))
    
    def _handle_voice_input(self, text):
        """处理语音识别结果"""
        if text == "请开始说话...":
            # 这是提示信息，不插入输入框
            self.statusBar().showMessage(text, 2000)
        else:
            # 将识别结果插入输入框
            current_text = self.input_box.toPlainText()
            if current_text and not current_text.endswith((' ', '\n')):
                self.input_box.insertPlainText(' ' + text)
            else:
                self.input_box.insertPlainText(text)
            self.statusBar().showMessage("语音输入完成", 2000)
    
    def _handle_voice_error(self, error_msg):
        """处理语音识别错误"""
        self.statusBar().showMessage(error_msg, 3000)
        QMessageBox.warning(self, "语音输入错误", error_msg)

    def _on_model_changed(self, model_name):
        self.api.model = model_name
        self.output_area.append(f"<b>已切换到模型:</b> {model_name}")
    
    def _clear_context(self):
        self.api.reset_context()
        self.output_area.clear()
        self.output_area.append("<b>对话历史已清除</b>")
    
    def _send_generate_message(self):
        prompt = self.input_box.toPlainText().strip()
        if not prompt:
            return
            
        self.output_area.append(f"<b>You:</b> {prompt}")
        self.input_box.clear()
        
        # 创建工作线程
        self.worker = GenerateWorker(self.api, prompt)
        self.worker.finished.connect(self._show_response)
        self.worker.error.connect(self._show_error)
        self.worker.start()
        
        # 禁用按钮防止重复发送
        self.send_btn.setEnabled(False)

    def _send_chat_message(self):
        prompt = self.input_box.toPlainText().strip()
        if not prompt:
            return
            
        self.output_area.append(f"<b>You:</b> {prompt}")
        self.input_box.clear()
        
        # 创建工作线程
        self.worker = ChatWorker(self.api, prompt)
        self.worker.finished.connect(self._show_response)
        self.worker.error.connect(self._show_error)
        self.worker.start()
        
        # 禁用按钮防止重复发送
        self.send_btn.setEnabled(False)

    def _show_response(self, response):

        # 处理Markdown转换
        html_response = markdown.markdown(response)
        self.current_response = html_response      
        self.output_area.append(f"<b>AI:</b> {html_response}")
        self.send_btn.setEnabled(True)
    
    def _show_error(self, error_msg):
        self.output_area.append(f"<b style='color:red'>错误:</b> {error_msg}")
        self.send_btn.setEnabled(True)

    def _toggle_mode(self):
        """切换聊天/生成模式"""
        self.is_chat_mode = not self.is_chat_mode
        
        # 更新模式显示
        if self.is_chat_mode:
            self.send_btn.clicked.disconnect(self._send_generate_message)
            self.send_btn.clicked.connect(self._send_chat_message)
        else:
            self.send_btn.clicked.disconnect(self._send_chat_message)
            self.send_btn.clicked.connect(self._send_generate_message)

        mode_text = "聊天模式" if self.is_chat_mode else "生成模式"
        self.mode_display.setText(mode_text)
    
            
        # 可选：添加状态栏提示
        self.statusBar().showMessage(f"已切换到{mode_text}", 2000)