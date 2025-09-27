from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QRadioButton, QGroupBox, QMessageBox , QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class ConfigPage(QWidget):
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        # เก็บค่าที่เลือกไว้ชั่วคราวจนกว่าจะกด Save
        self.selected_source = self.config_manager.camera_source

        # --- สร้าง Group Box ---
        camera_group_box = QGroupBox("Camera Source Selection")
        camera_group_box.setFont(QFont("Arial", 14))
        
        # --- สร้าง Radio Buttons ---
        self.radio_usb = QRadioButton("USB Camera")
        self.radio_hik = QRadioButton("Hikrobot Camera (Trigger Mode)")
        
        self.radio_usb.setFont(QFont("Arial", 12))
        self.radio_hik.setFont(QFont("Arial", 12))

        # --- ตั้งค่าเริ่มต้นตาม Config ที่โหลดมา ---
        if self.selected_source == "USB":
            self.radio_usb.setChecked(True)
        else:
            self.radio_hik.setChecked(True)

        # --- เชื่อมต่อสัญญาณเมื่อมีการเปลี่ยนแปลง ---
        self.radio_usb.toggled.connect(self.on_radio_button_toggled)
        self.radio_hik.toggled.connect(self.on_radio_button_toggled)

        # --- จัด Layout ภายใน Group Box ---
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.radio_usb)
        group_layout.addWidget(self.radio_hik)
        camera_group_box.setLayout(group_layout)

        # --- สร้างปุ่ม Save ---
        self.save_button = QPushButton("Save Settings")
        self.save_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.save_button.setFixedHeight(40)
        self.save_button.clicked.connect(self.save_settings)

        # --- จัด Layout หลักของหน้า ---
        main_layout = QVBoxLayout(self)
        title = QLabel("CONNECTION")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-size: 50px; font-weight: bold; color: #003366;")
        main_layout.addWidget(title)
        main_layout.addWidget(camera_group_box)
        main_layout.addWidget(self.save_button)
        main_layout.addStretch()
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        main_layout.setContentsMargins(20, 20, 20, 20)

    def on_radio_button_toggled(self):
        """
        อัปเดตค่าที่เลือกไว้ชั่วคราวเมื่อผู้ใช้เลือก Radio Button
        """
        if self.radio_usb.isChecked():
            self.selected_source = "USB"
        elif self.radio_hik.isChecked():
            self.selected_source = "Hikrobot"
        # ค่าจะยังไม่ถูกบันทึกลง ConfigManager จนกว่าจะกด Save

    def save_settings(self):
        """
        บันทึกค่าที่เลือกลงใน ConfigManager และไฟล์ .ini
        """
        # 1. อัปเดตค่าใน ConfigManager object
        self.config_manager.camera_source = self.selected_source
        
        # 2. สั่งให้ ConfigManager บันทึกค่าลงไฟล์
        self.config_manager.save_config()
        
        # 3. แสดง Popup ยืนยัน
        QMessageBox.information(self, "Settings Saved", "Camera settings have been saved successfully.")

