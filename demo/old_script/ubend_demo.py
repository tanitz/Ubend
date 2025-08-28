import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar, QRubberBand, QSizePolicy,
    QGridLayout
)
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtCore import Qt, QRect, QSize

class ScaledPixmapLabel(QLabel):
    """
    QLabel ที่ปรับขนาดรูปภาพให้พอดีกับพื้นที่และรักษาสัดส่วนของภาพต้นฉบับ
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap = QPixmap()

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_scaled_pixmap()

    def resizeEvent(self, event):
        self.update_scaled_pixmap()
        super().resizeEvent(event)

    def update_scaled_pixmap(self):
        if self._pixmap.isNull():
            super().setPixmap(QPixmap())
        else:
            scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled_pixmap)


class ImageLabel(ScaledPixmapLabel):
    """
    ScaledPixmapLabel ที่เพิ่มความสามารถในการ Crop ด้วยเมาส์
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setMouseTracking(True)
        self.rubber_band = None
        self.origin = None
        self.is_cropping_mode = False

    def set_cropping_mode(self, enabled):
        self.is_cropping_mode = enabled
        if not enabled and self.rubber_band:
            self.rubber_band.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_cropping_mode:
            self.origin = event.position().toPoint()
            if not self.rubber_band:
                self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if self.origin is not None and self.is_cropping_mode:
            self.rubber_band.setGeometry(QRect(self.origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_cropping_mode and self.rubber_band:
            self.main_window.finish_crop(self.rubber_band.geometry())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("โปรแกรมค้นหารูปภาพด้วย Template Matching")
        self.setGeometry(100, 100, 1280, 720)

        # --- ตัวแปรสำหรับจัดการสถานะ ---
        self.main_cv_img = None
        self.template = None
        self.target_image_paths = []
        self.current_target_index = -1
        
        # --- ค่า Default สำหรับขนาดภาพ ---
        self.DEFAULT_WIDTH = 800
        self.DEFAULT_HEIGHT = 600

        self.init_ui()
        self.show_default_frame() # แสดงกรอบเริ่มต้นเมื่อเปิดโปรแกรม

    def init_ui(self):
        page_layout = QHBoxLayout()
        
        # --- Panel ซ้าย: แสดงภาพหลัก ---
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("<b>Test Matching</b>"))
        self.image_label = ImageLabel(main_window=self)
        self.image_label.setStyleSheet("background-color: #2E2E2E; border: 1px solid #555;")
        left_panel.addWidget(self.image_label)
        
        # --- Panel ขวา: แสดงภาพที่หมุนและปุ่ม ---
        right_panel = QVBoxLayout()

        # ส่วนแสดงภาพที่หมุน
        right_top_panel = QVBoxLayout()
        right_top_panel.addWidget(QLabel("<b>Image Rotation</b>"))
        self.corrected_image_label = ScaledPixmapLabel()
        self.corrected_image_label.setStyleSheet("background-color: #2E2E2E; border: 1px solid #555;")
        right_top_panel.addWidget(self.corrected_image_label)
        
        # ส่วนปุ่มควบคุม (Grid Layout)
        button_grid = QGridLayout()
        self.btn_open = QPushButton("OPEN IMG")
        self.btn_crop = QPushButton("CROP")
        self.btn_prev_test = QPushButton("< TEST")
        self.btn_next_test = QPushButton("TEST >")

        button_grid.addWidget(self.btn_open, 0, 0)
        button_grid.addWidget(self.btn_crop, 0, 1)
        button_grid.addWidget(self.btn_prev_test, 1, 0)
        button_grid.addWidget(self.btn_next_test, 1, 1)

        for i in range(button_grid.count()):
            widget = button_grid.itemAt(i).widget()
            widget.setMinimumHeight(40)

        self.btn_crop.setCheckable(True)
        self.btn_open.clicked.connect(self.open_main_image)
        self.btn_crop.clicked.connect(self.toggle_crop_mode)
        self.btn_next_test.clicked.connect(self.test_next_image)
        self.btn_prev_test.clicked.connect(self.test_prev_image)

        right_panel.addLayout(right_top_panel, 65) # ให้ช่องภาพหมุนสูง 65% ของ panel ขวา
        right_panel.addLayout(button_grid, 35)      # ให้ปุ่มสูง 35% ของ panel ขวา

        page_layout.addLayout(left_panel, 70) # Panel ซ้ายกว้าง 70%
        page_layout.addLayout(right_panel, 30) # Panel ขวากว้าง 30%

        central_widget = QWidget()
        central_widget.setLayout(page_layout)
        self.setCentralWidget(central_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("กรุณาเปิดภาพต้นแบบเพื่อเริ่มต้น")

    def show_default_frame(self):
        """สร้างและแสดงกรอบสีดำเป็น Placeholder เริ่มต้น"""
        blank_image = np.zeros((self.DEFAULT_HEIGHT, self.DEFAULT_WIDTH, 3), np.uint8)
        self.display_image_on_label(blank_image, self.image_label)
        self.corrected_image_label.clear()
        self.update_status("กรุณาเปิดภาพต้นแบบเพื่อเริ่มต้น")

    def open_main_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกภาพต้นแบบ", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not path: return

        self.template = None
        self.current_target_index = -1
        self.btn_crop.setChecked(False)
        self.image_label.set_cropping_mode(False)

        self.main_cv_img = cv2.imread(path)
        if self.main_cv_img is None:
            self.update_status(f"Error: ไม่สามารถเปิดไฟล์ {os.path.basename(path)}")
            return
        
        # ปรับขนาดภาพต้นแบบตามค่า Default
        self.main_cv_img = cv2.resize(self.main_cv_img, (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))
            
        self.display_image_on_label(self.main_cv_img, self.image_label)
        self.corrected_image_label.clear()

        folder = os.path.dirname(path)
        current_file = os.path.basename(path)
        all_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.target_image_paths = [os.path.join(folder, f) for f in all_files if f != current_file]
        self.update_status(f"เปิดภาพ '{current_file}' | พบภาพอื่น {len(self.target_image_paths)} ภาพ")

    def toggle_crop_mode(self):
        if self.main_cv_img is None:
            self.update_status("กรุณาเปิดภาพต้นแบบก่อน")
            self.btn_crop.setChecked(False)
            return
            
        is_checked = self.btn_crop.isChecked()
        self.image_label.set_cropping_mode(is_checked)
        if is_checked:
            self.display_image_on_label(self.main_cv_img, self.image_label)
            self.corrected_image_label.clear()
            self.update_status("โหมด Crop: เปิดใช้งาน | ลากเมาส์บนภาพเพื่อเลือกพื้นที่")
        else:
            self.update_status("โหมด Crop: ปิดใช้งาน")

    def finish_crop(self, geometry):
        if self.main_cv_img is None: return
        
        current_pixmap = self.image_label.pixmap()
        if not current_pixmap or current_pixmap.isNull(): return

        orig_h, orig_w, _ = self.main_cv_img.shape
        pixmap_w = current_pixmap.width(); pixmap_h = current_pixmap.height()
        label_w = self.image_label.width(); label_h = self.image_label.height()

        offset_x = (label_w - pixmap_w) // 2; offset_y = (label_h - pixmap_h) // 2
        rb_x1 = geometry.left(); rb_y1 = geometry.top()
        rb_w = geometry.width(); rb_h = geometry.height()

        pix_x1 = rb_x1 - offset_x; pix_y1 = rb_y1 - offset_y
        scale_w = orig_w / pixmap_w; scale_h = orig_h / pixmap_h

        final_x1 = int(pix_x1 * scale_w); final_y1 = int(pix_y1 * scale_h)
        final_x2 = int((pix_x1 + rb_w) * scale_w); final_y2 = int((pix_y1 + rb_h) * scale_h)

        final_x1 = max(0, final_x1); final_y1 = max(0, final_y1)
        final_x2 = min(orig_w, final_x2); final_y2 = min(orig_h, final_y2)
        
        if final_x1 < final_x2 and final_y1 < final_y2:
            self.template = self.main_cv_img[final_y1:final_y2, final_x1:final_x2]
        else:
            self.template = None

        if self.template is not None and self.template.size > 0:
            self.update_status(f"สร้าง Template ขนาด {self.template.shape[1]}x{self.template.shape[0]} เรียบร้อยแล้ว")
            cv2.imshow("Template Preview", self.template)
        else:
            self.update_status("การ Crop ไม่สำเร็จ กรุณาลองใหม่")

        self.btn_crop.setChecked(False)
        self.image_label.set_cropping_mode(False)
        self.image_label.rubber_band.hide()

    def test_prev_image(self):
        if not self.target_image_paths: return
        self.current_target_index = (self.current_target_index - 1 + len(self.target_image_paths)) % len(self.target_image_paths)
        self.run_test_on_current_index()

    def test_next_image(self):
        if not self.target_image_paths: return
        self.current_target_index = (self.current_target_index + 1) % len(self.target_image_paths)
        self.run_test_on_current_index()

    def run_test_on_current_index(self):
        if self.template is None:
            self.update_status("กรุณาสร้าง Template โดยการ Crop ก่อน")
            return
        
        target_path = self.target_image_paths[self.current_target_index]
        target_img = cv2.imread(target_path)
        if target_img is None:
            self.update_status(f"Error: ไม่สามารถเปิดไฟล์ {os.path.basename(target_path)}")
            return
        
        # ปรับขนาดภาพเป้าหมายตามค่า Default
        target_img_resized = cv2.resize(target_img, (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))
            
        best_match = self.find_best_match(target_img_resized, self.template)
        
        display_img = target_img_resized.copy()
        if best_match['score'] >= 0.6:
            self.draw_match(display_img, best_match)
            
            h, w = best_match['template'].shape[:2]
            top_left = best_match['location']
            matched_area = target_img_resized[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
            
            corrected_image = self.rotate_image(matched_area, -best_match['angle'])
            self.display_image_on_label(corrected_image, self.corrected_image_label)
        else:
            self.corrected_image_label.clear()

        self.display_image_on_label(display_img, self.image_label)
        self.update_status(f"ผลการทดสอบกับ '{os.path.basename(target_path)}' | Score: {best_match['score']:.2f}")

    def find_best_match(self, target_img, template_img):
        best_match = {'score': -1, 'location': None, 'template': None, 'angle': 0}
        for angle in [0, 90, 180, 270]:
            rotated_template = self.rotate_image(template_img, angle)
            if rotated_template is None or \
               rotated_template.shape[0] > target_img.shape[0] or \
               rotated_template.shape[1] > target_img.shape[1]:
                continue
            result = cv2.matchTemplate(target_img, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_match['score']:
                best_match.update({'score': max_val, 'location': max_loc, 'template': rotated_template, 'angle': angle})
        return best_match

    def draw_match(self, image, match_data):
        h, w = match_data['template'].shape[:2]
        top_left = match_data['location']
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
        score_text = f"Match: {match_data['score']:.2f} (Angle: {match_data['angle']})"
        cv2.putText(image, score_text, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def display_image_on_label(self, cv_image, label_widget):
        if cv_image is None or cv_image.size == 0:
            label_widget.setPixmap(QPixmap())
            return
        h, w, ch = cv_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        label_widget.setPixmap(pixmap)

    def update_status(self, text):
        self.status_bar.showMessage(text)
        print(text)

    def rotate_image(self, image, angle):
        if image is None: return None
        (h, w) = image.shape[:2]
        if h == 0 or w == 0: return None
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos)); new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]; M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
