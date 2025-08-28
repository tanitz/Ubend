import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QSizePolicy,
    QFileDialog,
    QRubberBand,
    QTextEdit
)
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QPixmap, QImage, QPainter, QFont
import matching 

def convert_cv_to_pixmap(cv_image):
    """ฟังก์ชันสำหรับแปลงภาพจาก OpenCV (NumPy array) เป็น QPixmap"""
    if cv_image is None or cv_image.size == 0:
        return QPixmap()
    
    if len(cv_image.shape) == 2:
        h, w = cv_image.shape
        bytes_per_line = w
        qt_format = QImage.Format_Grayscale8
        rgb_image = cv_image
    else:
        h, w, ch = cv_image.shape
        if ch == 3:
            bytes_per_line = ch * w
            qt_format = QImage.Format_RGB888
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            return QPixmap()

    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, qt_format)
    return QPixmap.fromImage(qt_image)

class ImagePlaceholder(QLabel):
    """วิดเจ็ต QLabel แบบกำหนดเองที่รองรับการ Crop ภาพ"""
    def __init__(self, text, main_window, parent=None):
        super().__init__(text, parent)
        self.main_window = main_window
        self.original_cv_image = None
        self._pixmap = QPixmap()
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.crop_origin = QPoint()
        self.is_cropping_mode = False

        self.setStyleSheet("""
            background-color: black;
            color: white;
            border: 1px solid #555;
            font-size: 20px;
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_cv_image(self, cv_image, is_original=False):
        """ตั้งค่ารูปภาพจาก OpenCV array"""
        if is_original:
            self.original_cv_image = cv_image
        self.setPixmap(convert_cv_to_pixmap(cv_image))

    def setPixmap(self, pixmap):
        """เก็บ QPixmap และสั่งให้วาดใหม่"""
        self._pixmap = pixmap
        self.update()

    def set_cropping_mode(self, enabled):
        """เปิด/ปิด โหมดการ Crop"""
        self.is_cropping_mode = enabled
        if not enabled:
            self.rubber_band.hide()

    def get_crop_rect_on_original(self):
        """คำนวณและคืนค่าพื้นที่ที่เลือกบนภาพต้นฉบับ"""
        if self.original_cv_image is None or self._pixmap.isNull() or not self.rubber_band.isVisible():
            return None

        widget_crop_rect = self.rubber_band.geometry()
        scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_x_offset = (self.width() - scaled_pixmap.width()) / 2
        pixmap_y_offset = (self.height() - scaled_pixmap.height()) / 2
        
        crop_x_on_pixmap = widget_crop_rect.x() - pixmap_x_offset
        crop_y_on_pixmap = widget_crop_rect.y() - pixmap_y_offset

        if scaled_pixmap.height() == 0: return None
        scale = self.original_cv_image.shape[0] / scaled_pixmap.height()

        final_x = int(crop_x_on_pixmap * scale)
        final_y = int(crop_y_on_pixmap * scale)
        final_width = int(widget_crop_rect.width() * scale)
        final_height = int(widget_crop_rect.height() * scale)

        final_x = max(0, final_x)
        final_y = max(0, final_y)
        if final_x + final_width > self.original_cv_image.shape[1]:
            final_width = self.original_cv_image.shape[1] - final_x
        if final_y + final_height > self.original_cv_image.shape[0]:
            final_height = self.original_cv_image.shape[0] - final_y

        return QRect(final_x, final_y, final_width, final_height)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_cropping_mode:
            self.crop_origin = event.position().toPoint()
            self.rubber_band.setGeometry(QRect(self.crop_origin, QPoint()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if not self.crop_origin.isNull() and self.is_cropping_mode:
            self.rubber_band.setGeometry(QRect(self.crop_origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_cropping_mode:
            self.main_window.finish_crop()

    def paintEvent(self, event):
        if self._pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled_pixmap.width()) / 2
        y = (self.height() - scaled_pixmap.height()) / 2
        painter.drawPixmap(QPoint(x, y), scaled_pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 UI with Template Matching")
        
        self.main_cv_image = None
        self.template_cv_image = None
        self.test_image_paths = []
        self.current_test_index = -1
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # --- Layout หลักเป็นแนวนอน ---
        main_h_layout = QHBoxLayout(central_widget)

        # --- ส่วนหลักด้านซ้าย (ภาพและปุ่ม) ---
        left_container = QWidget()
        main_grid_layout = QGridLayout(left_container)

        # --- Layouts and Widgets ---
        left_v_layout = QVBoxLayout()
        self.image1 = ImagePlaceholder("IMAGE 1 (Test Image)", main_window=self)
        left_v_layout.addWidget(self.image1, 2)
        
        bottom_left_h_layout = QHBoxLayout()
        self.image2 = ImagePlaceholder("IMAGE 2 (Template)", main_window=self)
        self.image3 = ImagePlaceholder("IMAGE 3 (Rotated Result)", main_window=self)
        bottom_left_h_layout.addWidget(self.image2)
        bottom_left_h_layout.addWidget(self.image3)
        left_v_layout.addLayout(bottom_left_h_layout, 1)

        right_v_layout = QVBoxLayout()
        self.image4 = ImagePlaceholder("IMAGE 4 (Processed)", main_window=self)
        right_v_layout.addWidget(self.image4, 2)
        
        # --- ส่วนของปุ่มควบคุมหลัก ---
        button_group = QGroupBox("Controls")
        button_grid = QGridLayout()
        self.open_button = QPushButton("OPEN IMG")
        self.crop_button = QPushButton("CROP")
        self.save_template_button = QPushButton("SAVE TEMPLATE") # ปุ่มใหม่
        self.load_template_button = QPushButton("LOAD TEMPLATE") # ปุ่มใหม่
        self.prev_button = QPushButton("< TEST")
        self.next_button = QPushButton("TEST >")
        
        buttons = [self.open_button, self.crop_button, self.save_template_button, self.load_template_button, self.prev_button, self.next_button]
        for btn in buttons:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        button_grid.addWidget(self.open_button, 0, 0)
        button_grid.addWidget(self.crop_button, 0, 1)
        button_grid.addWidget(self.save_template_button, 1, 0)
        button_grid.addWidget(self.load_template_button, 1, 1)
        button_grid.addWidget(self.prev_button, 2, 0)
        button_grid.addWidget(self.next_button, 2, 1)
        button_group.setLayout(button_grid)
        right_v_layout.addWidget(button_group, 1)

        self.result_groupbox = QGroupBox("RESULT")
        self.result_layout = QVBoxLayout()
        self.result_label = QLabel("1. Open Image -> 2. Press CROP and drag -> 3. Use <TEST> / TEST>")
        self.result_layout.addWidget(self.result_label)
        self.result_groupbox.setLayout(self.result_layout)
        
        main_grid_layout.addLayout(left_v_layout, 0, 0)
        main_grid_layout.addLayout(right_v_layout, 0, 1)
        main_grid_layout.addWidget(self.result_groupbox, 1, 0, 1, 2)
        
        main_grid_layout.setRowStretch(0, 9)
        main_grid_layout.setRowStretch(1, 1)
        main_grid_layout.setColumnStretch(0, 1)
        main_grid_layout.setColumnStretch(1, 1)

        # --- ส่วนของ Code Editor ด้านขวา ---
        editor_group = QGroupBox("Code Editor")
        editor_layout = QVBoxLayout()
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Courier", 10))
        self.run_script_button = QPushButton("RUN SCRIPT")
        self.run_script_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        editor_layout.addWidget(self.code_editor)
        editor_layout.addWidget(self.run_script_button)
        editor_group.setLayout(editor_layout)

        # --- ประกอบ Layout หลัก ---
        main_h_layout.addWidget(left_container, 7)
        main_h_layout.addWidget(editor_group, 3)

        # --- เชื่อมต่อ Signals ---
        self.crop_button.setCheckable(True)
        self.open_button.clicked.connect(self.open_image)
        self.crop_button.toggled.connect(self.toggle_crop_mode)
        self.save_template_button.clicked.connect(self.save_template)
        self.load_template_button.clicked.connect(self.load_template)
        self.prev_button.clicked.connect(self.run_test_on_previous)
        self.next_button.clicked.connect(self.run_test_on_next)
        self.run_script_button.clicked.connect(self.process_image_from_3_to_4)

        self.set_default_editor_code()
        self.update_button_states()
        self.showMaximized()

    def set_default_editor_code(self):
        """ตั้งค่าโค้ดเริ่มต้นใน Code Editor"""
        default_code = (
            "# ภาพจาก IMAGE 3 จะถูกส่งเข้ามาในตัวแปร 'input_image'\n"
            "# ผลลัพธ์สุดท้ายต้องถูกเก็บไว้ในตัวแปร 'output_image'\n\n"
            "gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)\n"
            "output_image = cv2.Canny(gray, 100, 200)\n"
        )
        self.code_editor.setText(default_code)

    def update_button_states(self):
        """อัปเดตสถานะการเปิด/ปิดใช้งานของปุ่ม"""
        main_image_loaded = self.main_cv_image is not None
        template_created = self.template_cv_image is not None
        test_images_found = bool(self.test_image_paths)
        result_image_exists = self.image3.original_cv_image is not None

        self.crop_button.setEnabled(main_image_loaded)
        self.save_template_button.setEnabled(template_created)
        self.prev_button.setEnabled(template_created and test_images_found)
        self.next_button.setEnabled(template_created and test_images_found)
        self.run_script_button.setEnabled(result_image_exists)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not path: return

        try:
            with open(path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            self.main_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            self.main_cv_image = None
            self.result_label.setText(f"Error opening file: {e}")
        
        if self.main_cv_image is not None:
            self.image1.set_cv_image(self.main_cv_image, is_original=True)
            self.template_cv_image = None
            self.image2.set_cv_image(None); self.image2.setText("IMAGE 2 (Template)")
            self.image3.set_cv_image(None); self.image3.setText("IMAGE 3 (Rotated Result)")
            self.image4.set_cv_image(None); self.image4.setText("IMAGE 4 (Processed)")
            self.crop_button.setChecked(False)
            self.image1.set_cropping_mode(False)

            folder = os.path.dirname(path)
            current_file = os.path.basename(path)
            valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            all_files = sorted([f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in valid_extensions)])
            self.test_image_paths = [os.path.join(folder, f) for f in all_files if f != current_file]
            
            self.current_test_index = -1
            self.result_label.setText(f"Opened '{current_file}'. Found {len(self.test_image_paths)} other images to test.")

        self.update_button_states()

    def toggle_crop_mode(self, checked):
        if self.main_cv_image is None:
            self.crop_button.setChecked(False)
            return

        self.image1.set_cropping_mode(checked)
        if checked:
            self.result_label.setText("Crop mode ON. Drag on IMAGE 1 to create template.")
        else:
            self.result_label.setText("Crop mode OFF.")
            self.image1.rubber_band.hide()

    def finish_crop(self):
        crop_rect = self.image1.get_crop_rect_on_original()
        
        self.crop_button.setChecked(False)
        self.image1.set_cropping_mode(False)
        self.image1.rubber_band.hide()

        if not crop_rect or crop_rect.width() <= 0 or crop_rect.height() <= 0:
            self.result_label.setText("Crop cancelled or invalid area.")
            return

        self.template_cv_image = self.image1.original_cv_image[
            crop_rect.y():crop_rect.y() + crop_rect.height(),
            crop_rect.x():crop_rect.x() + crop_rect.width()
        ]
        self.image2.set_cv_image(self.template_cv_image, is_original=True)
        self.result_label.setText(f"Template created. Ready to test.")
        self.update_button_states()

    def save_template(self):
        if self.template_cv_image is None:
            self.result_label.setText("No template to save.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Template", "", "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)")
        if path:
            try:
                cv2.imwrite(path, self.template_cv_image)
                self.result_label.setText(f"Template saved to {os.path.basename(path)}")
            except Exception as e:
                self.result_label.setText(f"Error saving template: {e}")

    def load_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Template", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not path: return

        try:
            with open(path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            self.template_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            self.template_cv_image = None
            self.result_label.setText(f"Error loading template: {e}")

        if self.template_cv_image is not None:
            self.image2.set_cv_image(self.template_cv_image, is_original=True)
            self.result_label.setText(f"Template loaded from {os.path.basename(path)}. Ready to test.")
        
        self.update_button_states()

    def run_test_on_previous(self):
        if not self.test_image_paths: return
        self.current_test_index = (self.current_test_index - 1 + len(self.test_image_paths)) % len(self.test_image_paths)
        self.run_test_on_current_index()

    def run_test_on_next(self):
        if not self.test_image_paths: return
        self.current_test_index = (self.current_test_index + 1) % len(self.test_image_paths)
        self.run_test_on_current_index()

    def run_test_on_current_index(self):
        if self.template_cv_image is None:
            self.result_label.setText("Cannot test: Please create a template by cropping first.")
            return
        
        target_path = self.test_image_paths[self.current_test_index]
        try:
            with open(target_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            target_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            self.result_label.setText(f"Error opening {os.path.basename(target_path)}: {e}")
            return
        
        if target_img is not None:
            self.perform_rotation_matching(target_img, self.template_cv_image)

    def perform_rotation_matching(self, target_image, template):
        import cv2, numpy as np, time, math
        from PySide6.QtWidgets import QApplication

        start_t = time.time()
        TIMEOUT = 12.0

        if template is None or template.size == 0:
            self.result_label.setText("No template.")
            return

        src = target_image.copy()
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        th, tw = template_gray.shape

        matches = matching.match_template_rotated_coarse_to_fine(
            target_gray, template_gray,
            coarse_step=1,  # coarse step finding
            fine_step=1,
            coarse_range=(-360, 360), # angle search range
            fine_window=2,  # step search in best angle
            candidates_per_angle=1, # increase number for detecting multiple object at same angle
            min_score = 0.5, # minimum score to be considered a match
            scale = 0.1,  # Scale is big, cost time, more accuracy & vise-versa
            iou_thresh = 0.2 # overlay ratio
        )
        found_crop = False
        for i, match in enumerate(matches):
            pos = match["position"]
            angle = match["angle"]
            score = match["score"]
            # print(pos, angle)
            target_result = matching.draw_rotated_match(src, pos, tw, th, angle, label=f"#{i} {angle:.1f}° {score:.2f}")
            rotated_src, rot_mat = matching.rotate_image(target_image, angle, border_value=255)
            center_in_rot = matching.apply_affine_transform(pos, rot_mat)
            cx_r, cy_r = int(round(center_in_rot[0])), int(round(center_in_rot[1]))
            x0 = cx_r - tw // 2
            y0 = cy_r - th // 2
            x1 = x0 + tw
            y1 = y0 + th
            pad_left = max(0, -x0)
            pad_top = max(0, -y0)
            pad_right = max(0, x1 - rotated_src.shape[1])
            pad_bottom = max(0, y1 - rotated_src.shape[0])

            crop_x0 = max(0, x0)
            crop_y0 = max(0, y0)
            crop_x1 = min(rotated_src.shape[1], x1)
            crop_y1 = min(rotated_src.shape[0], y1)
            if crop_x1 > crop_x0 and crop_y1 > crop_y0:
                crop = rotated_src[crop_y0:crop_y1, crop_x0:crop_x1]
                if pad_top or pad_bottom or pad_left or pad_right:
                    crop = cv2.copyMakeBorder(
                        crop,
                        top=pad_top, bottom=pad_bottom,
                        left=pad_left, right=pad_right,
                        borderType=cv2.BORDER_CONSTANT,
                        value=255
                    )
                self.image3.set_cv_image(crop, is_original=True)
                save_dir = "/home/snc/Documents/Ubend/machine_vision/image_rotated"
                os.makedirs(save_dir, exist_ok=True)
                save_name = f"rotated_{time.strftime('%Y%m%d_%H%M%S')}_{i}.png"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, crop)
                found_crop = True
                self.update_button_states() 
                break  # ใช้ match แรกที่เจอ
        if not found_crop:
            self.image3.set_cv_image(None)
            self.update_button_states() 
        self.image1.set_cv_image(target_result, is_original=True)
        QApplication.processEvents()

    def process_image_from_3_to_4(self):
        """นำโค้ดจาก Editor มาประมวลผลภาพจากช่อง 3 และแสดงในช่อง 4"""
        input_image = self.image3.original_cv_image
        if input_image is None:
            self.result_label.setText("No image in Result panel to process.")
            return

        code_to_run = self.code_editor.toPlainText()
        
        local_scope = {
            'cv2': cv2,
            'np': np,
            'input_image': input_image.copy()
        }
        
        try:
            exec(code_to_run, {}, local_scope)
            
            output_image = local_scope.get('output_image')

            if output_image is not None:
                self.image4.set_cv_image(output_image, is_original=True)
                self.result_label.setText("Script executed successfully.")
            else:
                self.result_label.setText("Script ran, but 'output_image' variable not found.")

        except Exception as e:
            self.result_label.setText(f"Script Error: {e}")


if __name__ == "__main__":
    # --- เพิ่มไฟล์เริ่มต้น ---
    default_image_path = "/home/snc/Documents/Ubend/machine_vision/image_demo/image_10.jpg"
    default_template_path = "/home/snc/Documents/Ubend/machine_vision/image_demo/Template.png"

    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()

    # --- โหลดภาพหลักและเทมเพลตทันทีเมื่อเริ่มโปรแกรม ---
    if os.path.exists(default_image_path):
        try:
            with open(default_image_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            window.main_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            window.image1.set_cv_image(window.main_cv_image, is_original=True)
        except Exception as e:
            window.result_label.setText(f"Error opening default image: {e}")

    if os.path.exists(default_template_path):
        try:
            with open(default_template_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            window.template_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            window.image2.set_cv_image(window.template_cv_image, is_original=True)
            window.result_label.setText("Template loaded automatically. Ready to test.")
        except Exception as e:
            window.result_label.setText(f"Error opening default template: {e}")

    # --- เพิ่มการค้นหา test images อัตโนมัติและรันเทสทันที ---
    if window.main_cv_image is not None:
        folder = os.path.dirname(default_image_path)
        current_file = os.path.basename(default_image_path)
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        all_files = sorted([f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in valid_extensions)])
        window.test_image_paths = [os.path.join(folder, f) for f in all_files if f != current_file]
        window.current_test_index = 0 if window.test_image_paths else -1

    window.update_button_states()

    # --- รันเทสทันทีถ้ามีเทมเพลตและ test image ---
    if window.template_cv_image is not None and window.test_image_paths:
        window.run_test_on_current_index()

    sys.exit(app.exec())