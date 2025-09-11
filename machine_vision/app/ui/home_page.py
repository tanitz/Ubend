import sys
import os
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSizePolicy, QGroupBox ,QGridLayout
)
from PySide6.QtCore import Qt, QPoint, QRect, QEvent, QTimer , Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QFont
from PySide6.QtWidgets import QApplication
import app.matching
from app.camera_workers import USBCameraWorker, HikrobotCameraWorker
import configparser
from dataclasses import dataclass
import app.config_manager as cfg_mgr

@dataclass
class Config:
    camera_source: str = "Hikrobot"
    # เพิ่ม field อื่นๆ ตามต้องการ

def _load_config() -> Config:
    cfg = Config()
    # คาดว่า config.ini อยู่ที่ repository root (machine_vision/config.ini)
    ini_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.ini"))
    parser = configparser.ConfigParser()
    try:
        if os.path.exists(ini_path):
            parser.read(ini_path)
            # อ่าน section/option ตามโครงไฟล์ของคุณ
            if parser.has_section("camera"):
                cfg.camera_source = parser.get("camera", "source", fallback=cfg.camera_source)
    except Exception:
        # ป้องกัน crash ถ้าไฟล์มีปัญหา
        pass
    return cfg

# โหลด .ini หนึ่งครั้งเมื่อ module นี้ถูก import
CONFIG = _load_config()

class ImagePlaceholder(QLabel):
    """QLabel that stores original cv image and supports rubber-band style crop."""
    cropFinished = Signal() 
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.original_cv_image = None
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #111; color: #ddd; border:1px solid #444;")
        self.rubber_origin = QPoint()
        self.rubber_rect = QRect()
        self.is_cropping = False

    def set_cv_image(self, cv_image, is_original=False):
        if cv_image is None:
            self.original_cv_image = None if is_original else self.original_cv_image
            self._pixmap = QPixmap()
            super().setPixmap(self._pixmap)
            return
        if is_original:
            # store original full-res image so we can rescale on resize
            self.original_cv_image = cv_image.copy()

        src = cv_image if not is_original else self.original_cv_image

        # convert to QPixmap from src (use original when available to avoid multiple conversions)
        if src is None:
            self._pixmap = QPixmap()
            super().setPixmap(self._pixmap)
            return

        if len(src.shape) == 2:
            h, w = src.shape
            bytes_per_line = w
            qimg = QImage(src.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            h, w, ch = src.shape
            rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # ถ้าขนาด widget ยังไม่ถูกตั้ง ให้เก็บ pixmap เต็มความละเอียดไว้ก่อน
        target_w = max(1, self.width())
        target_h = max(1, self.height())
        # scale ให้เต็มพื้นที่โดยรักษาอัตราส่วน
        scaled = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._pixmap = scaled
        super().setPixmap(self._pixmap)

    def resizeEvent(self, event):
        """เมื่อ widget ถูกย่อ/ขยาย ให้สเกลภาพจาก original_cv_image ใหม่เสมอ."""
        super().resizeEvent(event)
        if self.original_cv_image is not None:
            # ใช้ original_cv_image เพื่อได้คุณภาพดีที่สุดเมื่อสเกลใหม่
            try:
                self.set_cv_image(self.original_cv_image, is_original=True)
            except Exception:
                # ป้องกัน crash ถ้าเกิดขณะ resize
                pass

    def set_cropping_mode(self, enabled: bool):
        self.is_cropping = enabled
        if not enabled:
            self.rubber_rect = QRect()
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_cropping:
            self.rubber_origin = event.pos()
            self.rubber_rect = QRect(self.rubber_origin, self.rubber_origin)
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_cropping and not self.rubber_origin.isNull():
            self.rubber_rect = QRect(self.rubber_origin, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_cropping:
            self.is_cropping = False
            self.update()
            # notify parent (assume parent widget has finish_crop)
            # if hasattr(self.parent(), "finish_crop"):
            #     self.parent().finish_crop()
            self.cropFinished.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap.isNull():
            return
        if self.rubber_rect and not self.rubber_rect.isNull():
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rubber_rect)

    def get_crop_rect_on_original(self):
        """Map rubber_rect (widget coords) to original_cv_image coords."""
        if self.original_cv_image is None or self._pixmap.isNull() or self.rubber_rect.isNull():
            return None
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_off = int((self.width() - scaled.width()) / 2)
        y_off = int((self.height() - scaled.height()) / 2)
        rx = max(0, self.rubber_rect.x() - x_off)
        ry = max(0, self.rubber_rect.y() - y_off)
        rw = min(scaled.width(), self.rubber_rect.width())
        rh = min(scaled.height(), self.rubber_rect.height())
        if rw <= 0 or rh <= 0:
            return None
        scale = self.original_cv_image.shape[1] / scaled.width()  # width-based scale
        ox = int(rx * scale)
        oy = int(ry * scale)
        ow = int(rw * scale)
        oh = int(rh * scale)
        ox = max(0, ox); oy = max(0, oy)
        if ox + ow > self.original_cv_image.shape[1]:
            ow = self.original_cv_image.shape[1] - ox
        if oy + oh > self.original_cv_image.shape[0]:
            oh = self.original_cv_image.shape[0] - oy
        if ow <= 0 or oh <= 0:
            return None
        return QRect(ox, oy, ow, oh)


class HomePage(QWidget):
    """Main page: uses Hikrobot (or USB) camera as source and provides matching UI."""
    oring_status = Signal(str, str)
    emit_summary = Signal(str)
    def __init__(self, config_manager=None):
        super().__init__()
        # ใช้ config_manager ที่ส่งมา ถ้าเป็น None ให้ใช้ singleton ที่โหลด .ini เมื่อเริ่มโปรแกรม
        self.config_manager = config_manager if config_manager is not None else cfg_mgr.CONFIG_MANAGER
        self.is_camera_on = False
        self.camera_worker = None
        self.cv_frame = None
        self.cropped_cv_frame = None
        self.template_cv_image = None
        self.crop_mode = False

        # ======================================================================
        # 1. สร้าง Widgets ทั้งหมด
        # ======================================================================

        # Widgets แสดงผลภาพ
        self.image1 = ImagePlaceholder("IMAGE 1 (Live)", self)
        self.image2 = ImagePlaceholder("IMAGE 2 (Template)", self)
        self.image3 = ImagePlaceholder("IMAGE 3 (Cropped Result)", self)
        self.image4 = ImagePlaceholder("IMAGE 4 (Processed)", self)
        self.image1.cropFinished.connect(self.finish_crop)

        

        # ตั้งค่า Style และการแสดงผล
        for w in (self.image1, self.image2, self.image3, self.image4):
            w.setStyleSheet("background-color:#111; border:1px solid #444; color:#fff;")
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # <--- ใช้ Expanding
            w.setScaledContents(False) 

        # Widgets ส่วนควบคุม (Controls)
        self.start_button = QPushButton("START CAMERA")
        self.crop_button = QPushButton("CROP")
        self.save_template_button = QPushButton("SAVE TEMPLATE")
        self.run_match_button = QPushButton("RUN")
        # ปุ่มจากตัวอย่างแรก (ถ้าต้องการใช้)
        # self.load_template_button = QPushButton("LOAD TEMPLATE")

        # ตั้งค่าปุ่ม
        self.crop_button.setCheckable(True)
        self.save_template_button.setEnabled(False)
        self.run_match_button.setCheckable(True)           # ทำปุ่มเป็น toggle
        self.run_match_button.setEnabled(False)

        # ปรับขนาดและสไตล์ปุ่มให้ใหญ่ขึ้น
        btn_font = QFont()
        btn_font.setPointSize(11)  # ปรับขนาดตัวอักษร
        for b in (self.start_button, self.crop_button, self.save_template_button, self.run_match_button):
            b.setFont(btn_font)
            b.setMinimumHeight(44)                  # ความสูงขั้นต่ำของปุ่ม
            b.setMinimumWidth(140)                  # ความกว้างขั้นต่ำ (ถ้าต้องการ)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet("padding:8px;")         # เพิ่มช่องว่างภายในปุ่ม

        # Widget แถบสถานะ (Status bar)
        self.result_label = QLabel("Ready")
        self.result_label.setStyleSheet("background-color:#222; color:#ffd; padding:6px;")


        # ======================================================================
        # 2. จัดวาง Layouts
        # ======================================================================

        # --- ส่วนซ้าย ---
        image1_group = QGroupBox("RAW IMAGE")
        image1_layout = QVBoxLayout()
        image1_layout.addWidget(self.image1)
        image1_group.setLayout(image1_layout)

        # สร้าง GroupBox สำหรับ image2 และ image3
        self.image2_group = QGroupBox("MATCHING TEMPLATE")
        image2_layout = QVBoxLayout()
        image2_layout.addWidget(self.image2)
        self.image2_group.setLayout(image2_layout)

        self.image3_group = QGroupBox("MATCHING RESULT")
        image3_layout = QVBoxLayout()
        image3_layout.addWidget(self.image3)
        self.image3_group.setLayout(image3_layout)
        
        

        bottom_left_h_layout = QHBoxLayout()
        bottom_left_h_layout.addWidget(self.image2_group)
        bottom_left_h_layout.addWidget(self.image3_group)

        left_v_layout = QVBoxLayout()
        # กำหนดสัดส่วนให้ image1 ใหญ่กว่าส่วนล่าง (2:1)
        left_v_layout.addWidget(image1_group, stretch=4)
        left_v_layout.addLayout(bottom_left_h_layout, stretch=2)

        # --- ส่วนขวา ---
        # สร้าง GroupBox สำหรับ Controls
        controls_group = QGroupBox("CONTROLS")
        # ใช้ GridLayout จัดเรียงปุ่มให้เป็นตาราง 2x2
        controls_grid_layout = QGridLayout()
        controls_grid_layout.addWidget(self.start_button, 0, 0)
        controls_grid_layout.addWidget(self.crop_button, 0, 1)
        controls_grid_layout.addWidget(self.save_template_button, 1, 0)
        controls_grid_layout.addWidget(self.run_match_button, 1, 1)
        controls_group.setLayout(controls_grid_layout)
        
        right_v_layout = QVBoxLayout()
        # สร้าง GroupBox สำหรับ image4
        self.image4_group = QGroupBox("PROCESSED RESULT")
        image4_layout = QVBoxLayout()
        image4_layout.addWidget(self.image4)
        self.image4_group.setLayout(image4_layout)

        self.result_group = QGroupBox("RESULT")
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)
        self.result_group.setLayout(result_layout)

        # กำหนดสัดส่วนให้ image4 ใหญ่กว่าส่วน Controls (2:1)
        right_v_layout.addWidget(self.image4_group, stretch=4)
        right_v_layout.addWidget(self.result_group, stretch=1)
        right_v_layout.addWidget(controls_group, stretch=1)
        

        # --- Layout หลัก (ซ้าย-ขวา) ---
        main_h_layout = QHBoxLayout()
        # กำหนดสัดส่วนให้ฝั่งซ้ายกว้างกว่าฝั่งขวาเล็กน้อย (3:2)
        main_h_layout.addLayout(left_v_layout, stretch=2)
        main_h_layout.addLayout(right_v_layout, stretch=2)
        
        # --- Layout รวมทั้งหมด (บน-ล่าง) ---
        overall_v_layout = QVBoxLayout(self)  # ใช้ self เพื่อบอกว่านี่คือ layout หลักของหน้าต่าง
        title = QLabel("DASHBOARD")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #003366;")
        overall_v_layout.addWidget(title)
        overall_v_layout.addLayout(main_h_layout)
        # # สร้าง GroupBox สำหรับผลลัพธ์ (RESULT)
        # self.result_group = QGroupBox("RESULT")
        # result_layout = QVBoxLayout()
        # result_layout.addWidget(self.result_label)
        # self.result_group.setLayout(result_layout)
        # overall_v_layout.addWidget(self.result_group)  # เพิ่มแถบสถานะด้านล่างในกล่อง

        # ======================================================================
        # 3. เชื่อมต่อ Signals
        # ======================================================================
        self.start_button.clicked.connect(self.toggle_camera)
        self.crop_button.toggled.connect(self.toggle_crop_mode)
        self.save_template_button.clicked.connect(self.save_template)
        self.run_match_button.toggled.connect(self.toggle_auto_match)
        # timer เพื่อรัน matching เป็น loop โดยไม่บล็อก UI
        self.auto_match_timer = QTimer(self)
        self.auto_match_timer.setInterval(200)  # ms, ปรับตามต้องการ
        self.auto_match_timer.timeout.connect(self._auto_match_tick)
        self.matching_running = False
        self.destroyed.connect(self.cleanup)

        # load default template from machine_vision/temp.jpg (if exists)
        try:
            default_template_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "temp.jpg")
            )
            if os.path.exists(default_template_path):
                tpl = cv2.imread(default_template_path)
                if tpl is not None:
                    self.template_cv_image = tpl
                    self.image2.set_cv_image(self.template_cv_image, is_original=True)
                    self.save_template_button.setEnabled(True)
                    self.run_match_button.setEnabled(True)
                    self.result_label.setText(f"Template loaded: {os.path.basename(default_template_path)}")
        except Exception:
            pass

    def toggle_camera(self):
        if not self.is_camera_on:
            source = getattr(self.config_manager, "camera_source", "Hikrobot")
            if source == "USB":
                self.camera_worker = USBCameraWorker()
            else:
                self.camera_worker = HikrobotCameraWorker()
            self.camera_worker.frame_ready.connect(self.update_image_slot)
            self.camera_worker.error_occurred.connect(self.show_camera_error)
            self.camera_worker.start()
            self.is_camera_on = True
            self.start_button.setText("STOP CAMERA")
            self.result_label.setText("Starting camera...")
        else:
            if self.camera_worker:
                self.camera_worker.stop()
                self.camera_worker.wait()
                self.camera_worker = None
            self.is_camera_on = False
            self.start_button.setText("START CAMERA")
            self.result_label.setText("Stopped")

    def show_camera_error(self, msg):
        self.result_label.setText(f"Camera error: {msg}")
        self.is_camera_on = False
        self.start_button.setText("START CAMERA")

    def update_image_slot(self, cv_img: np.ndarray):
        """Slot called by worker signal (GUI thread)."""
        if cv_img is None:
            return
        self.cv_frame = cv_img
        self.image1.set_cv_image(self.cv_frame, is_original=True)
        if self.cropped_cv_frame is not None:
            self.image3.set_cv_image(self.cropped_cv_frame, is_original=True)

    def toggle_crop_mode(self, checked):
        if not self.is_camera_on or self.cv_frame is None:
            self.crop_button.setChecked(False)
            return
        self.image1.set_cropping_mode(checked)
        self.result_label.setText("Crop mode ON" if checked else "Crop mode OFF")

    def finish_crop(self):
        rect = self.image1.get_crop_rect_on_original()
        self.image1.set_cropping_mode(False)
        self.crop_button.setChecked(False)
        if rect is None:
            self.result_label.setText("Invalid crop")
            return
        self.template_cv_image = self.image1.original_cv_image[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()].copy()
        # update template preview and force repaint/layout so it's visible inside the QGroupBox
        self.image2.set_cv_image(self.template_cv_image, is_original=True)
        # ensure widget and its group are repainted and layouts updated
        try:
            self.image2.update()
            self.image2.repaint()
            if hasattr(self, "image2_group"):
                self.image2_group.updateGeometry()
                self.image2_group.repaint()
            QApplication.processEvents(
)
        except Exception:
            pass
        self.save_template_button.setEnabled(True)
        self.run_match_button.setEnabled(True)
        self.result_label.setText("Template created")

    def save_template(self):
        if self.template_cv_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Template", "", "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)")
        if not path:
            return
        cv2.imwrite(path, self.template_cv_image)
        self.result_label.setText(f"Template saved: {os.path.basename(path)}")

    def run_match_on_current_frame(self):
        # ป้องกันเรียกซ้อนถ้ามีการรันอยู่แล้ว
        if self.matching_running:
            return
        self.matching_running = True
        timings = {}
        total_start = time.perf_counter()
        try:
            if self.template_cv_image is None:
                self.result_label.setText("No template")
                return
            if self.cv_frame is None:
                self.result_label.setText("No live frame")
                return
            
            # rotation_matching
            t0 = time.perf_counter()
            crop = self.rotation_matching(self.cv_frame, self.template_cv_image)
            timings['rotation_matching'] = (time.perf_counter() - t0) * 1000.0

            if crop is None:
                self.result_label.setText("No match found")
                return

            self.cropped_cv_frame = crop

            # update image3
            t1 = time.perf_counter()
            self.image3.set_cv_image(crop, is_original=True)
            timings['set_image3'] = (time.perf_counter() - t1) * 1000.0

            # find_oring
            t2 = time.perf_counter()
            final_img = self.find_oring(crop)
            timings['find_oring'] = (time.perf_counter() - t2) * 1000.0

            if final_img is None:
                self.result_label.setText("final_img not found")
                return

            # update image4
            t3 = time.perf_counter()
            self.image4.set_cv_image(final_img, is_original=True)
            timings['set_image4'] = (time.perf_counter() - t3) * 1000.0

            total_ms = (time.perf_counter() - total_start) * 1000.0
            # สรุปเวลา (แสดง millisecond)
            summary = (f"Total {total_ms:.1f}ms | ")
            # summary = (f"Total {total_ms:.1f}ms | "
            #            f"rot:{timings.get('rotation_matching',0):.1f}ms "
            #            f"find:{timings.get('find_oring',0):.1f}ms "
            #            f"img3:{timings.get('set_image3',0):.1f}ms "
            #            f"img4:{timings.get('set_image4',0):.1f}ms")
            self.result_label.setText(summary)

        finally:
            self.matching_running = False
            self.cv_frame = None

    def toggle_auto_match(self, checked: bool):
        """เริ่ม/หยุด loop การ match โดยใช้ timer"""
        if checked:
            if not self.is_camera_on or self.cv_frame is None:
                self.run_match_button.setChecked(False)
                self.run_match_button.setText("RUN")
                self.result_label.setText("Camera not ready")
                return
            self.auto_match_timer.start()
            self.run_match_button.setText("STOP")
            self.result_label.setText("Auto-match ON")
        else:
            self.auto_match_timer.stop()
            self.run_match_button.setText("RUN")
            self.result_label.setText("Auto-match OFF")

    def _auto_match_tick(self):
        """เรียก matching ถ้าไม่มีการรันอยู่ และมีเฟรมใหม่"""
        if self.matching_running:
            return
        if self.cv_frame is None or self.template_cv_image is None:
            return
        # เรียกโดยตรง (rotation_matching ทำการอัพเดต UI ภายใน)
        self.run_match_on_current_frame()

    def rotation_matching(self, target_image, template):

        start_t = time.time()
        TIMEOUT = 12.0

        if template is None or template.size == 0:
            self.result_label.setText("No template.")
            return

        src = target_image.copy()
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        th, tw = template_gray.shape

        matches = app.matching.match_template_rotated_coarse_to_fine(
            target_gray, template_gray,
            coarse_step=6,  # coarse step finding
            fine_step=0.1,
            coarse_range=(-180, 180), # angle search range
            fine_window=0.1,  # step search in best angle
            candidates_per_angle=3, # increase number for detecting multiple object at same angle
            min_score = 0.5, # minimum score to be considered a match
            scale = 0.08,  # Scale is big, cost time, more accuracy & vise-versa
            iou_thresh = 0.2 # overlay ratio
        )
        found_crop = False
        crop = None  # ป้องกันกรณีไม่เจอ match แล้ว return ตัวแปรยังไม่ถูกกำหนด
        for i, match in enumerate(matches):
            pos = match["position"]
            angle = match["angle"]
            score = match["score"]
            if -80.0 < angle < -75.0:
                angle = angle-6
                #pos offset x -2
                pos = (pos[0], pos[1]+20)
            if -133.0 < angle < -130.0:
                #angle = angle-6
                #pos offset x -2
                pos = (pos[0]+40, pos[1]+50)
            if 105.0 < angle < 110.0:
                angle = angle-10
                #pos offset x -2
                pos = (pos[0]+20, pos[1]-20)
            # print(pos, angle)
            target_result = app.matching.draw_rotated_match(src, pos, tw, th, angle, label=f"#{i} {angle:.1f}° {score:.2f}")
            rotated_src, rot_mat = app.matching.rotate_image(target_image, angle, border_value=255)
            center_in_rot = app.matching.apply_affine_transform(pos, rot_mat)
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
                break  # ใช้ match แรกที่เจอ
        if not found_crop and crop is None:
            self.image3.set_cv_image(None)

        self.image1.set_cv_image(target_result, is_original=True)
        QApplication.processEvents()
        return crop

    def find_oring(self, image: np.ndarray,
                   large_area: int = 5000,
                   dif_zone: int = 0,
                   h_oring: int = 65,
                   w_oring: int = 110) -> np.ndarray:
        """Simplified find_oring adapted from matching_check.py"""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        # read main params from config_manager (allow override by explicit args)
        large_area = int(getattr(self.config_manager, "large_area", large_area)) if large_area == 5000 else int(large_area)
        dif_zone = int(getattr(self.config_manager, "dif_zone", dif_zone)) if dif_zone == 0 else int(dif_zone)
        h_oring = int(getattr(self.config_manager, "h_oring", h_oring)) if h_oring == 65 else int(h_oring)
        w_oring = int(getattr(self.config_manager, "w_oring", w_oring)) if w_oring == 110 else int(w_oring)

        # read image-processing params from config_manager
        blur_kernel = int(getattr(self.config_manager, "blur_kernel", 15))
        # ensure odd kernel size
        if blur_kernel % 2 == 0:
            blur_kernel = max(3, blur_kernel - 1)
        canny_thresh1 = int(getattr(self.config_manager, "canny_thresh1", 50))
        canny_thresh2 = int(getattr(self.config_manager, "canny_thresh2", 90))
        morph_kernel = int(getattr(self.config_manager, "morph_kernel", 15))
        if morph_kernel <= 0:
            morph_kernel = 15
        try:
            rect_height_ratio = float(getattr(self.config_manager, "rect_height_ratio", 0.5))
        except Exception:
            rect_height_ratio = 0.5

        img = image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        # note: original used v and named s_increased; keep behaviour but clarity:
        channel = v
        s_increased = cv2.add(channel, 10)
        _, object_mask = cv2.threshold(s_increased, 0, 100, cv2.THRESH_BINARY)
        _, hole_mask = cv2.threshold(s_increased, 30, 255, cv2.THRESH_BINARY_INV)
        inpaint_mask = cv2.bitwise_and(object_mask, hole_mask)
        try:
            _ = cv2.inpaint(s_increased, inpaint_mask, 1, cv2.INPAINT_NS)
        except Exception:
            pass
        blurred = cv2.GaussianBlur(inpaint_mask, (blur_kernel, blur_kernel), 0)
        _, smoothed = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(smoothed, canny_thresh1, canny_thresh2)
        contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(cv2.contourArea(contours2[0]))
        # DEBUG: print counts (จะพิมพ์เฉพาะเมื่อเปิด env UBEND_DEBUG=1)
        try:
            if os.environ.get("UBEND_DEBUG"):
                print("find_oring: contours total=", len(contours2), "large_area=", large_area)
        except Exception:
            pass

        large_contours = [c for c in contours2 if cv2.contourArea(c) >= large_area]

        # Fallback: ถ้าไม่มี large_contours ให้ลองลดเกณฑ์และลอง morphology เพื่อช่วยเชื่อมรูปร่าง
        if not large_contours and contours2:
            # ใช้ top few contours แทน (ลดเกณฑ์)
            small_sorted = sorted(contours2, key=cv2.contourArea, reverse=True)[:4]
            large_contours = small_sorted

        if not large_contours:
            # morphology fallback
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            closed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
            edges2 = cv2.Canny(closed, max(10, canny_thresh1 - 10), min(255, canny_thresh2 + 10))
            contours2b, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [c for c in contours2b if cv2.contourArea(c) >= large_area]

            try:
                if os.environ.get("UBEND_DEBUG"):
                    print("find_oring: fallback contours after morph=", len(contours2b), "large_after=", len(large_contours))
                    cv2.imwrite("/tmp/oring_smoothed.png", smoothed)
                    cv2.imwrite("/tmp/oring_edges.png", edges)
                    cv2.imwrite("/tmp/oring_edges2.png", edges2)
            except Exception:
                pass

        overlay = img.copy()
        if large_contours:
            cv2.drawContours(overlay, large_contours, -1, (0, 200, 0), 5)
        p = cv2.addWeighted(img, 0.7, overlay, 0.5, 0)
        hh, ww = p.shape[:2]
        x_mid = ww // 2
        y_mid = int(np.clip(hh // 2 - dif_zone, 0, hh))
        font = cv2.FONT_HERSHEY_SIMPLEX
        zone_status = {}
        pts = None
        if large_contours:
            try:
                pts = np.vstack(large_contours)[:, 0, :].astype(np.int32)
            except Exception:
                pts = None
        zones = {3: (0, x_mid, y_mid, hh), 4: (x_mid, ww, y_mid, hh)}
        for z, (x0, x1, y0, y1) in zones.items():
            if pts is None or pts.size == 0:
                continue
            mask_zone = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
            pts_zone = pts[mask_zone]
            if pts_zone.size == 0:
                continue
            x_min = int(np.min(pts_zone[:, 0])); x_max = int(np.max(pts_zone[:, 0]))
            y_max = int(np.max(pts_zone[:, 1]))
            y_at_x_min = int(np.median(pts_zone[pts_zone[:, 0] == x_min][:, 1])) if np.any(pts_zone[:, 0] == x_min) else int(np.median(pts_zone[:, 1]))
            y_at_x_max = int(np.median(pts_zone[pts_zone[:, 0] == x_max][:, 1])) if np.any(pts_zone[:, 0] == x_max) else int(np.median(pts_zone[:, 1]))
            avg_y = (y_at_x_min + y_at_x_max) // 2
            rect_width = x_max - x_min
            rect_height = max(1, int(rect_height_ratio * rect_width))
            top_y = max(0, avg_y - rect_height // 2)
            bottom_y = min(hh - 1, avg_y + rect_height // 2)
            diff_val = abs(avg_y - y_max)
            fill_color = (0, 255, 0) if (diff_val > h_oring and rect_width > w_oring) else (0, 0, 255)
            text_result = "OK" if (diff_val > h_oring and rect_width > w_oring) else "NG"
            zone_status[z] = text_result
            overlay2 = p.copy()
            cv2.rectangle(overlay2, (x_min, top_y), (x_max, bottom_y), fill_color, -1)
            p = cv2.addWeighted(overlay2, 0.4, p, 0.6, 0)
            cv2.rectangle(p, (x_min, top_y), (x_max, bottom_y), (255, 0, 0), 1)
        s1 = zone_status.get(3, "NA").upper()
        s2 = zone_status.get(4, "NA").upper()
        summary = f"oring1 {s1.lower()} | oring2 {s2.lower()}"
        if s1 == "OK" and s2 == "OK":
            top_color = (0, 255, 0)
        elif s1 == "NG" or s2 == "NG":
            top_color = (0, 0, 255)
        else:
            top_color = (0, 255, 255)
        cv2.putText(p, summary, (10, 30), font, 0.8, top_color, 2, cv2.LINE_AA)
        try:
            self.oring_status.emit(s1, s2)
        except Exception:
            pass
        try:
            self.emit_summary.emit(summary)
        except Exception:
            pass
        try:
            self.result_label.setText(summary)
        except Exception:
            pass
        return p

    def cleanup(self):
        if self.auto_match_timer.isActive():
            self.auto_match_timer.stop()
            self.run_match_button.setChecked(False)
            self.run_match_button.setText("RUN")
        if self.camera_worker and self.camera_worker.isRunning():
             self.camera_worker.stop()
             self.camera_worker.wait()
             self.camera_worker = None
