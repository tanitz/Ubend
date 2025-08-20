# setting_page.py
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt

class SettingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel("นี่คือหน้า Setting")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)