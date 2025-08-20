import sys
import os
import subprocess
import cv2
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, 
    QFileDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QFont

# --- Import Worker จากไฟล์ใหม่ ---
from app.camera_workers import USBCameraWorker, HikrobotCameraWorker

class ImageSelectionLabel(QLabel):
    def __init__(self, parent_widget=None):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.selection_rect = QRect()
        self.origin_point = QPoint()
        self.is_selecting = False

    def mousePressEvent(self, event):
        if self.parent_widget.is_crop_mode and event.button() == Qt.LeftButton:
            self.is_selecting = True
            self.origin_point = event.pos()
            self.selection_rect = QRect(self.origin_point, self.origin_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.selection_rect = QRect(self.origin_point, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_selecting and event.button() == Qt.LeftButton:
            self.is_selecting = False
            self.parent_widget.set_crop_area(self.selection_rect)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.parent_widget.is_crop_mode and self.is_selecting:
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)


class HomePage(QWidget):
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager # รับ Config Manager เข้ามา
        self.is_camera_on = False
        self.is_crop_mode = False
        self.cv_frame = None
        self.cropped_cv_frame = None
        self.crop_image_rect = None
        self.camera_worker = None

        # --- สร้างวิดเจ็ต (เหมือนเดิม) ---
        self.raw_image_boxui = ImageSelectionLabel(self)
        self.crop_image_boxui = QLabel("IMAGE CROP REAL TIME")
        self.template_label = QLabel()

        for label in [self.raw_image_boxui, self.crop_image_boxui, self.template_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid #555; background-color: #333; color: white;")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setScaledContents(True)

        self.template_label.setFixedSize(160, 120)

        self.start_stop_button = QPushButton("START CAMERA")
        self.crop_button = QPushButton("CROP")
        self.save_button = QPushButton("SAVE")
        
        for button in [self.start_stop_button, self.crop_button, self.save_button]:
            button.setFixedHeight(40)
            button.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.crop_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # --- จัด Layout (เหมือนเดิม) ---
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.raw_image_boxui, 1)
        left_layout.addWidget(self.crop_image_boxui, 1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.template_label, 0, Qt.AlignCenter)
        right_layout.addWidget(self.start_stop_button)
        right_layout.addWidget(self.crop_button)
        right_layout.addWidget(self.save_button)
        right_layout.addStretch()

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(left_layout, 7)
        main_layout.addLayout(right_layout, 3)

        # --- สัญญาณและการเชื่อมต่อ (เหมือนเดิม) ---
        self.start_stop_button.clicked.connect(self.toggle_camera)
        self.crop_button.clicked.connect(self.enter_crop_mode)
        self.save_button.clicked.connect(self.save_image)

    def toggle_camera(self):
        if not self.is_camera_on:
            # --- 🎯 เลือก Worker ตามการตั้งค่า ---
            source = self.config_manager.camera_source
            print(f"Starting camera with source: {source}")
            if source == "USB":
                self.camera_worker = USBCameraWorker()
            elif source == "Hikrobot":
                self.camera_worker = HikrobotCameraWorker()
            else:
                self.show_camera_error(f"ไม่รู้จัก Camera Source: {source}")
                return

            self.camera_worker.frame_ready.connect(self.update_image_slot)
            self.camera_worker.error_occurred.connect(self.show_camera_error)
            self.camera_worker.start()
            
            self.is_camera_on = True
            self.raw_image_boxui.setText("Starting camera...")
            self.start_stop_button.setText("STOP CAMERA")
            self.crop_button.setEnabled(True)
        else:
            if self.camera_worker:
                self.camera_worker.stop()
            self.is_camera_on = False
            self.raw_image_boxui.setText("IMAGE RAW REAL TIME")
            self.crop_image_boxui.setText("IMAGE CROP REAL TIME")
            self.template_label.clear()
            self.start_stop_button.setText("START CAMERA")
            self.crop_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.crop_image_rect = None

    def show_camera_error(self, message):
        self.raw_image_boxui.setText(f"Error: {message}")
        self.is_camera_on = False
        self.start_stop_button.setText("START CAMERA")
        self.crop_button.setEnabled(False)

    def update_image_slot(self, cv_img):
        self.cv_frame = cv_img
        self.display_image(self.cv_frame, self.raw_image_boxui)
        if self.crop_image_rect:
            x, y, w, h = self.crop_image_rect
            self.cropped_cv_frame = self.cv_frame[y:y+h, x:x+w]
            self.display_image(self.cropped_cv_frame, self.crop_image_boxui)

    def enter_crop_mode(self):
        if not self.is_camera_on or self.cv_frame is None: return
        self.is_crop_mode = True
        self.raw_image_boxui.setCursor(Qt.CrossCursor)

    def set_crop_area(self, selection_rect):
        self.is_crop_mode = False
        self.raw_image_boxui.setCursor(Qt.ArrowCursor)
        if self.cv_frame is None: return
        label_size = self.raw_image_boxui.size()
        frame_h, frame_w, _ = self.cv_frame.shape
        scale_x = frame_w / label_size.width()
        scale_y = frame_h / label_size.height()
        x = int(selection_rect.x() * scale_x)
        y = int(selection_rect.y() * scale_y)
        w = int(selection_rect.width() * scale_x)
        h = int(selection_rect.height() * scale_y)
        x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
        if x + w > frame_w: w = frame_w - x
        if y + h > frame_h: h = frame_h - y
        if w > 0 and h > 0:
            self.crop_image_rect = (x, y, w, h)
            template_frame = self.cv_frame[y:y+h, x:x+w]
            self.display_image(template_frame, self.template_label)
            self.save_button.setEnabled(True)

    def display_image(self, cv_img, label):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image))

    def save_image(self):
        if self.cropped_cv_frame is None: return
        file_path, _ = QFileDialog.getSaveFileName(self, "บันทึกภาพ", "", "Images (*.png *.jpg)")
        if file_path:
            cv2.imwrite(file_path, self.cropped_cv_frame)
            print(f"Image saved to: {file_path}")
            try:
                if sys.platform == "win32": os.startfile(file_path)
                elif sys.platform == "darwin": subprocess.run(["open", file_path])
                else: subprocess.run(["xdg-open", file_path])
            except Exception as e:
                print(f"Could not open file automatically: {e}")
