from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, 
    QTextBrowser, QTextEdit, QPushButton,
    QHBoxLayout, QLabel, QComboBox, QMessageBox, QFileDialog, QInputDialog, QLineEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import pyqtSlot
from threads.worker import GenerateWorker, ChatWorker
from threads.streaming_worker import StreamingWorker
from threads.voice_input import VoskVoiceInputThread
from core.langchain_ollama_client import LangchainOllamaAPI
import markdown  # 用于处理Markdown格式
import pyttsx3  # 添加语音合成库
import re
import os
import shutil
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
        self.api = LangchainOllamaAPI()
        
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
        
        # 添加文件上传按钮
        self.upload_btn = QPushButton("上传文件")
        self.upload_btn.clicked.connect(self._upload_file)
        self.upload_btn.setToolTip("上传文档到知识库")

        control_layout.addWidget(QLabel("选择模型:"))
        control_layout.addWidget(self.model_combo, 3)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.upload_btn)
        
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
        self.chat_mode = 0
        self.mode_display = QLabel("生成模式")

        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self._send_generate_message)

        # self.mode_display.setStyleSheet("font-weight: bold; color: #4ec9b0;")
        self.change_mode_btn = QPushButton("切换模式")
        self.change_mode_btn.clicked.connect(self._toggle_mode)
        # self.change_mode_btn.setEnabled(False)

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
        response_list = self.api.get_model_list()
        for m in response_list.models:
            if m.model == "nomic-embed-text:latest":
                pass
            else:
                self.model_combo.addItem(m.model)
    
    def _speak_output(self):
        """朗读输出区域的内容"""
        try:
            # 获取输出区域的纯文本内容（去除HTML标签）
            plain_text = self.current_response
            # print(f"朗读内容: {plain_text}")  # 调试输出
            plain_text = re.sub(r'<[^>]+>', '', plain_text)  # 移除所有HTML标签
            # print(f"朗读内容: {plain_text}")  # 调试输出

            plain_text = re.sub(r'[^\w\s,.!?！？，。]', '', plain_text)
            # print(f"朗读内容: {plain_text}")  # 调试输出

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
        self.api.change_model(model_name)
        self.output_area.append(f"<b>已切换到模型:</b> {model_name}")
        self.statusBar().showMessage(f"已切换到模型: {model_name}", 2000)
        self._clear_context()
    
    def _clear_context(self):
        self.api.reset_context()
        self.current_response = ""  # 清空当前响应
        self.output_area.clear()
        self.output_area.append("<b>对话历史已清除</b>")
    
    @pyqtSlot()
    def _send_generate_message_stream(self):
        prompt = self.input_box.toPlainText().strip()
        if not prompt:
            return
            
        # 禁用发送按钮防止重复发送
        self.send_btn.setEnabled(False)
        
        # 显示用户消息
        self._append_user_message(prompt)
        
        # 清空输入框
        self.input_box.clear()
        
        # 重置当前响应
        self.current_response = ""
        
        # 显示初始的"思考中..."消息
        self._append_ai_message("思考中...")

        # 创建工作线程
        self.worker = StreamingWorker(self.api, prompt)
        self.worker.partial_response.connect(self._update_partial_response)
        self.worker.finished.connect(self._on_stream_finished)
        self.worker.error.connect(self._show_stream_error)
        self.worker.start()
        
        # 禁用按钮防止重复发送
        self.send_btn.setEnabled(False)

    @pyqtSlot(str)
    def _update_partial_response(self, response):
        """更新部分响应 - 替换最后一条AI消息"""
        # 移除之前的"思考中..."消息
        self._remove_last_ai_message()
        
        # 更新当前响应
        self.current_response = response
        # 显示新内容
        self._append_ai_message(response)

    @pyqtSlot()
    def _on_stream_finished(self):
        """流式处理完成"""
        # 确保最后一条消息是最终结果（而不是"思考中..."）
        self._remove_last_ai_message()
        # print(f"最终响应内容: {self.current_response}")  # 调试输出
        plain_text = re.sub(r'<[^>]+>', '', self.current_response)  # 移除所有HTML标签
        # print(f"最终响应内容: {plain_text}")  # 调试输出
        plain_text = re.sub(r'[^\w\s,.!?！？，。]', '', plain_text)
        # print(f"最终响应内容: {plain_text}")  # 调试输出
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()  # 合并多余空格
        self.current_response = plain_text  # 更新当前响应为纯文本
        print(f"最终响应内容: {self.current_response}")  # 调试输出
        self._append_ai_message(self.current_response)
        
        # 清理资源
        self.worker = None
        self.current_response = ""
        self.send_btn.setEnabled(True)
    
    @pyqtSlot(str)
    def _show_stream_error(self, error_msg):
        """显示错误信息"""
        self._remove_last_ai_message()
        self._append_ai_message(f"<span style='color: red;'>{error_msg}</span>")
        self._on_stream_finished()

    def _append_user_message(self, text):
        """添加用户消息"""
        self.output_area.append(f"""
            <div style='
                margin: 10px;
                background-color: #e3f2fd;
                border-radius: 10px;
                padding: 10px;
                margin-left: 60px;
                text-align: right;
            '>
                <b>你:</b> {text}
            </div>
        """)
        self._scroll_to_bottom()

    def _append_ai_message(self, text):
        """添加AI消息"""
        html = f"""
            <div style='margin-bottom: 15px;'>
                <div style='
                    margin: 10px;
                    background-color: #ffffff;
                    border-radius: 10px;
                    padding: 10px;
                    margin-right: 60px;
                    text-align: left;
                '>
                    <b>AI:</b> {text}
                </div>
            </div>
        """
        self.output_area.append(html)
        self._scroll_to_bottom()
    
    def _remove_last_ai_message(self):
        cursor = self.output_area.textCursor()
        # 移动到文档末尾
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # 选择当前块（也就是最后一条消息所在的块）
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        # 删除选中的块
        cursor.removeSelectedText()

    def _scroll_to_bottom(self):
        """滚动到底部"""
        scrollbar = self.output_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

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
        if (self.chat_mode==0 ):
            self.send_btn.clicked.disconnect(self._send_generate_message)
            self.chat_mode = 1
            mode_text = "聊天模式"
            self.send_btn.clicked.connect(self._send_chat_message)
        elif (self.chat_mode==1):
            self.send_btn.clicked.disconnect(self._send_chat_message)
            self.chat_mode = 2
            mode_text = "检索模式"
            self.send_btn.clicked.connect(self._send_generate_message_stream)
        elif (self.chat_mode==2):
            self.send_btn.clicked.disconnect(self._send_generate_message_stream)
            self.chat_mode = 0
            mode_text = "生成模式"
            self.send_btn.clicked.connect(self._send_generate_message)
        else:
            raise ValueError("未知模式")

        self.mode_display.setText(mode_text)  
        self._clear_context()
        # 可选：添加状态栏提示
        self.statusBar().showMessage(f"已切换到{mode_text}", 2000)

    def _upload_file(self):
        """打开文件对话框选择文件并上传到知识库（处理文件已存在情况）"""
        # 支持的文件类型
        file_types = "文档文件 (*.pdf *.docx *.doc *.txt)"
        
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择要上传的文档", 
            "", 
            file_types
        )
        
        if not file_path:
            return  # 用户取消了选择
        
        # 获取文件名
        file_name = os.path.basename(file_path)
        
        # 目标目录
        target_dir = self.api.get_documentes_dir()
        
        # 确保目标目录存在
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # 目标路径
        target_path = os.path.join(target_dir, file_name)
        
        try:
            # 检查文件是否已存在
            if os.path.exists(target_path):
                # 询问用户如何处理
                reply = QMessageBox.question(
                    self,
                    "文件已存在",
                    f"文件 '{file_name}' 已存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel
                )
                
                
                if reply == QMessageBox.StandardButton.Cancel:
                    self.output_area.append(
                        f"<div style='color: orange;'>已取消上传: 文件 '{file_name}' 已存在</div>"
                    )
                    return
                elif reply == QMessageBox.StandardButton.No:
                    # 获取新文件名
                    new_name, ok = QInputDialog.getText(
                        self,
                        "重命名文件",
                        "输入新文件名:",
                        QLineEdit.EchoMode.Normal,
                        file_name
                    )
                    
                    if not ok or not new_name:
                        self.output_area.append(
                            f"<div style='color: orange;'>已取消上传: 未提供新文件名</div>"
                        )
                        return
                    
                    # 更新目标路径
                    target_path = os.path.join(target_dir, new_name)
                    file_name = new_name  # 更新用于显示的文件名
            
            # 复制文件到目标目录
            shutil.copyfile(file_path, target_path)
            
            # 更新索引
            self.api.rebuild_index_and_chain()
            
            # 显示成功消息
            self.output_area.append(
                f"<div style='color: green;'>已成功上传文档: {file_name}</div>"
            )
            
            # 滚动到底部
            self.output_area.verticalScrollBar().setValue(
                self.output_area.verticalScrollBar().maximum()
            )
        except Exception as e:
            # 显示错误消息
            self.output_area.append(
                f"<div style='color: red;'>上传失败: {str(e)}</div>"
            )
