from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class IoPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel("This is I/O Page")
        label.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 24)
        label.setFont(font)
        layout.addWidget(label)
