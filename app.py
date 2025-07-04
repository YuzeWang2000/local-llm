import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import ChatWindow

def main():
    app = QApplication(sys.argv)
    
    # # 设置全局样式
    # with open('resources/styles/main.qss', 'r') as f:
    #     app.setStyleSheet(f.read())
    
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()