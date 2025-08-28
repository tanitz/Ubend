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
    QRubberBand
)
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QPixmap, QImage, QPainter

def convert_cv_to_pixmap(cv_image):
    """Helper function to convert an OpenCV image (NumPy array) to a QPixmap."""
    if cv_image is None:
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
    """
    A custom QLabel that supports interactive cropping and stores the OpenCV image.
    It handles mapping crop areas from the scaled view to the original image.
    """
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.original_cv_image = None # The clean, unmodified image
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
        """Sets the image from an OpenCV image array."""
        if is_original:
            self.original_cv_image = cv_image
        self.setPixmap(convert_cv_to_pixmap(cv_image))

    def setPixmap(self, pixmap):
        """Stores the pixmap and triggers a repaint."""
        self._pixmap = pixmap
        self.update()

    def set_cropping_mode(self, enabled):
        """Enable or disable the cropping mode."""
        self.is_cropping_mode = enabled
        if not enabled:
            self.rubber_band.hide()

    def get_crop_rect_on_original(self):
        """Returns the selected rectangle mapped to original image coordinates."""
        if self.original_cv_image is None or self._pixmap.isNull() or not self.rubber_band.isVisible():
            return None

        widget_crop_rect = self.rubber_band.geometry()
        scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_x_offset = (self.width() - scaled_pixmap.width()) / 2
        pixmap_y_offset = (self.height() - scaled_pixmap.height()) / 2
        
        crop_x_on_pixmap = widget_crop_rect.x() - pixmap_x_offset
        crop_y_on_pixmap = widget_crop_rect.y() - pixmap_y_offset

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
            # FIX: Use event.position().toPoint() instead of deprecated event.pos()
            self.crop_origin = event.position().toPoint()
            self.rubber_band.setGeometry(QRect(self.crop_origin, QPoint()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if not self.crop_origin.isNull() and self.is_cropping_mode:
            # FIX: Use event.position().toPoint() instead of deprecated event.pos()
            self.rubber_band.setGeometry(QRect(self.crop_origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_cropping_mode:
            pass # Selection is finalized, user will click the CROP button

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
        self.resize(1280, 720)

        self.main_cv_image = None # This will hold the image used for cropping the template
        self.template_cv_image = None
        self.test_image_paths = []
        self.current_test_index = -1
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout(central_widget)

        # --- Layouts and Widgets ---
        left_v_layout = QVBoxLayout()
        self.image1 = ImagePlaceholder("IMAGE 1 (Test/Template Source)")
        left_v_layout.addWidget(self.image1, 2)
        
        bottom_left_h_layout = QHBoxLayout()
        self.image2 = ImagePlaceholder("IMAGE 2 (Template)")
        self.image3 = ImagePlaceholder("IMAGE 3 (Result)")
        bottom_left_h_layout.addWidget(self.image2)
        bottom_left_h_layout.addWidget(self.image3)
        left_v_layout.addLayout(bottom_left_h_layout, 1)

        right_v_layout = QVBoxLayout()
        self.image4 = ImagePlaceholder("IMAGE 4")
        right_v_layout.addWidget(self.image4, 2)
        
        button_grid = QGridLayout()
        self.open_button = QPushButton("OPEN TEMPLATE IMG")
        self.crop_button = QPushButton("CROP")
        self.prev_button = QPushButton("< TEST")
        self.next_button = QPushButton("TEST >")
        for btn in [self.open_button, self.crop_button, self.prev_button, self.next_button]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button_grid.addWidget(self.open_button, 0, 0)
        button_grid.addWidget(self.crop_button, 0, 1)
        button_grid.addWidget(self.prev_button, 1, 0)
        button_grid.addWidget(self.next_button, 1, 1)
        right_v_layout.addLayout(button_grid, 1)

        self.result_groupbox = QGroupBox("RESULT")
        self.result_layout = QVBoxLayout()
        self.result_label = QLabel("1. Open Template Image -> 2. Crop -> 3. Use <TEST> / TEST>")
        self.result_layout.addWidget(self.result_label)
        self.result_groupbox.setLayout(self.result_layout)
        
        main_layout.addLayout(left_v_layout, 0, 0)
        main_layout.addLayout(right_v_layout, 0, 1)
        main_layout.addWidget(self.result_groupbox, 1, 0, 1, 2)
        main_layout.setRowStretch(0, 4)
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        
        # --- Connect Signals ---
        self.crop_button.setCheckable(True)
        self.open_button.clicked.connect(self.open_template_image)
        self.crop_button.clicked.connect(self.toggle_crop_mode)
        self.prev_button.clicked.connect(self.run_test_on_previous)
        self.next_button.clicked.connect(self.run_test_on_next)

        self.update_button_states()

    def update_button_states(self):
        """Enable/disable buttons based on application state."""
        main_image_loaded = self.main_cv_image is not None
        template_created = self.template_cv_image is not None
        test_images_found = bool(self.test_image_paths)

        self.crop_button.setEnabled(main_image_loaded)
        self.prev_button.setEnabled(template_created and test_images_found)
        self.next_button.setEnabled(template_created and test_images_found)

    def open_template_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Template Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
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
            self.template_cv_image = None # Reset template
            self.image2.set_cv_image(None)
            self.image2.setText("IMAGE 2 (Template)")
            self.image3.set_cv_image(None)
            self.image3.setText("IMAGE 3 (Result)")
            self.crop_button.setChecked(False)
            self.image1.set_cropping_mode(False)

            # Find other images in the same folder to use as test targets
            folder = os.path.dirname(path)
            current_file = os.path.basename(path)
            valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            all_files = sorted([f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in valid_extensions)])
            self.test_image_paths = [os.path.join(folder, f) for f in all_files if f != current_file]
            
            self.current_test_index = -1
            self.result_label.setText(f"Opened '{current_file}'. Found {len(self.test_image_paths)} other images to test.")

        self.update_button_states()

    def toggle_crop_mode(self, checked):
        """Activates or deactivates the cropping mode."""
        if self.main_cv_image is None:
            self.crop_button.setChecked(False)
            return

        self.image1.set_cropping_mode(checked)
        if checked:
            self.result_label.setText("Crop mode ON. Drag on IMAGE 1 to select, then click CROP again to confirm.")
        else:
            # This is where the crop is finalized
            self.finish_crop()

    def finish_crop(self):
        """Finalizes the crop and creates the template."""
        crop_rect = self.image1.get_crop_rect_on_original()
        if not crop_rect or crop_rect.width() <= 0 or crop_rect.height() <= 0:
            self.result_label.setText("Crop cancelled or invalid area.")
            return

        self.template_cv_image = self.main_cv_image[
            crop_rect.y():crop_rect.y() + crop_rect.height(),
            crop_rect.x():crop_rect.x() + crop_rect.width()
        ]
        self.image2.set_cv_image(self.template_cv_image)
        self.result_label.setText(f"Template created ({self.template_cv_image.shape[1]}x{self.template_cv_image.shape[0]}). Ready to test.")
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
            self.image1.set_cv_image(target_img, is_original=True) # Display the target image
            self.perform_rotation_matching(target_img, self.template_cv_image)

    def perform_rotation_matching(self, main_image, template):
        self.result_label.setText("Matching in progress...")
        QApplication.processEvents()

        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template_gray.shape

        best_score, best_angle, best_loc = -1, 0, None
        
        for angle in range(0, 360, 5):
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            cos, sin = np.abs(rot_mat[0, 0]), np.abs(rot_mat[0, 1])
            new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
            
            rot_mat[0, 2] += (new_w / 2) - center[0]
            rot_mat[1, 2] += (new_h / 2) - center[1]
            
            rotated_template = cv2.warpAffine(template_gray, rot_mat, (new_w, new_h))
            
            if rotated_template.shape[0] > main_gray.shape[0] or rotated_template.shape[1] > main_gray.shape[1]:
                continue

            res = cv2.matchTemplate(main_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score, best_loc, best_angle = max_val, max_loc, angle
        
        threshold = 0.6
        if best_score >= threshold:
            self.result_label.setText(f"Match Found! Score: {best_score:.4f} at {best_angle}°")
            
            result_img_1 = main_image.copy()
            rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
            box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
            pts = cv2.transform(np.array([box]), rot_mat)[0]
            pts += np.array(best_loc)
            cv2.polylines(result_img_1, [np.int32(pts)], True, (0, 255, 0), 3, cv2.LINE_AA)
            self.image1.set_cv_image(result_img_1)

            center_of_found_rect = (best_loc[0] + pts[1][0]/2, best_loc[1] + pts[2][1]/2)
            inv_rot_mat = cv2.getRotationMatrix2D(center_of_found_rect, -best_angle, 1.0)
            straight_result = cv2.warpAffine(main_image, inv_rot_mat, (main_image.shape[1], main_image.shape[0]))
            final_crop = cv2.getRectSubPix(straight_result, (w, h), center_of_found_rect)
            self.image3.set_cv_image(final_crop)
        else:
            self.result_label.setText(f"No match found. Best score: {best_score:.4f}")
            self.image1.set_cv_image(main_image, is_original=True) # Show original image without box
            self.image3.set_cv_image(None)
            self.image3.setText("IMAGE 3 (No Match)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
