from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class ReportPage(QWidget):
    """Simple Report page (unlocked). Extend later with real reporting UI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        title = QLabel("REPORT")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #003366;")

        hint = QLabel("This is the report page. Add charts, tables, or logs here.")
        hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        hint.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addStretch(1)
