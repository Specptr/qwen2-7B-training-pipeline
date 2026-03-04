# app.py
# 26.3.4
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit,
    QLineEdit, QPushButton, QVBoxLayout, QGridLayout,
    QHBoxLayout, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTextCursor
from infer import generate_stream

PHOTO_PATH = r"assets/S.png"

from PySide6.QtCore import QThread, Signal

class ModelThread(QThread):
    token_signal = Signal(str)

    def __init__(self, prompt, history):
        super().__init__()
        self.prompt = prompt
        self.history = history

    def run(self):
        for token in generate_stream(self.prompt, self.history):
            self.token_signal.emit(token)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("")
        self.resize(800, 500)
        self.history = []

        self.setStyleSheet("""
            QWidget {
                background-color: #050505;
                font-family: "Segoe UI";
                color: #e0e0e0;
                font-size: 16px;
            }

            QFrame {
                background-color: #000000;
                border: 1.5px solid #c1c1c1;
                border-radius: 16px;
            }

            QTextEdit {
                background-color: #050505;
                border: none;
                padding: 20px;
            }

            QLineEdit {
                background-color: #000000;
                border: none;
                padding: 14px;
            }

            QPushButton {
                background-color: #000000;
                border-radius: 10px;
                border: 2px solid #c0c0c0;
                padding: 8px 14px;
            }

            QPushButton:hover {
                background-color: #1a1a1a;
            }

            QLineEdit {
                padding-left: 16px;
            }

            QPushButton {
                min-width: 36px;
                max-width: 36px;
                height: 36px;
            }

            #no_frame {
                border: none;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 20, 30, 20)

        # ===== 标题 =====
        title = QLabel("Sereia")
        title.setStyleSheet("font-size: 15px; letter-spacing: 2px;")
        title.setAlignment(Qt.AlignLeft)
        title.setContentsMargins(10, 0, 0, 0)
        title.setObjectName("no_frame")
        main_layout.addWidget(title)

        # ===== 中间区域 =====
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(30)

        # 左侧照片区
        self.photo_frame = QFrame()
        photo_layout = QVBoxLayout()
        photo_layout.setContentsMargins(0, 0, 0, 0)

        self.photo_label = QLabel()
        self.photo_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(PHOTO_PATH)
        if not pixmap.isNull():
            self.photo_label.setPixmap(
                pixmap.scaled(
                    250, 400,
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation
                )
            )
        else:
            self.photo_label.setText("Her")

        photo_layout.addWidget(self.photo_label)
        self.photo_frame.setLayout(photo_layout)
        self.photo_frame.setFixedWidth(240)

        # 右侧对话区
        self.chat_frame = QFrame()
        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)

        chat_layout.addWidget(self.chat_area)
        self.chat_frame.setLayout(chat_layout)

        middle_layout.addWidget(self.photo_frame)
        middle_layout.addWidget(self.chat_frame)

        main_layout.addLayout(middle_layout)

        # ===== 底部区域 =====
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(40)

       # 左侧按钮区
        self.control_frame = QFrame()
        control_layout = QGridLayout()
        control_layout.setSpacing(15)

        btn1 = QPushButton("reset")
        btn2 = QPushButton("clear")
        btn3 = QPushButton("kiss")

        control_layout.addWidget(btn1, 0, 0)
        control_layout.addWidget(btn2, 0, 1)
        control_layout.addWidget(btn3, 0, 2)

        self.control_frame.setLayout(control_layout)
        self.control_frame.setFixedWidth(240)
        self.control_frame.setObjectName("no_frame")

        # 右侧输入区
        self.input_frame = QFrame()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(10)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("say something...")

        self.send_btn = QPushButton("⁜")
        self.send_btn.setFixedHeight(40)

        self.send_btn.clicked.connect(self.send_message)
        self.input_box.returnPressed.connect(self.send_message)

        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_btn)

        self.input_frame.setLayout(input_layout)
        self.input_frame.setFixedHeight(70)

        bottom_layout.addWidget(self.control_frame)
        bottom_layout.addWidget(self.input_frame)

        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def send_message(self):
        text = self.input_box.text().strip()
        if not text:
            return

        self.chat_area.append(f"\n你：{text}\n")
        self.input_box.clear()

        self.chat_area.append("她：")

        self.thread = ModelThread(text, self.history)
        self.thread.token_signal.connect(self.append_token)
        self.thread.start()

    def append_token(self, token):
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.chat_area.setTextCursor(cursor)
        self.chat_area.ensureCursorVisible()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
