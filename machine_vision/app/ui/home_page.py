import sys
import os
import configparser
import time
import csv
from datetime import datetime
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSizePolicy, QGroupBox ,QGridLayout
)
from PySide6.QtCore import Qt, QPoint, QRect, QEvent, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QFont
from PySide6.QtWidgets import QApplication
import app.matching
try:
    demo_match_path = "/home/snc/Documents/DEMO/opencv_matching"
    if demo_match_path not in sys.path:
        sys.path.append(demo_match_path)
    import matching2 as tm_matching   # wrapper for native matcher
except Exception as _e:
    tm_matching = None
    print("WARNING: matching2 import failed:", _e)
from app.camera_workers import USBCameraWorker, HikrobotCameraWorker
import configparser
from dataclasses import dataclass
import app.config_manager as cfg_mgr
import socket   # <-- add
import json     # <-- add

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
    classification_emitted = Signal(str) 
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
        # ---- counters ----
        self.count_ok = 0
        self.count_ng = 0
        self.count_na = 0
        self.count_total = 0

        self._matcher = None
        self._template_size = None  # (w,h)

        # TCP settings
        self.tcp_enabled = False
        self.tcp_host = "127.0.0.1"
        self.tcp_port = 5000
        self.tcp_mode = "plain"   # plain | json
        self._load_tcp_config()
        # เชื่อม signal ส่ง TCP
        

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
        self.crop_button = QPushButton("MATCHING CROP")
        self.open_template_button = QPushButton("OPEN TEMPLATE")
        self.run_match_button = QPushButton("RUN")
        # ปุ่มจากตัวอย่างแรก (ถ้าต้องการใช้)
        # self.load_template_button = QPushButton("LOAD TEMPLATE")

        # ตั้งค่าปุ่ม
        self.crop_button.setCheckable(True)
        self.open_template_button.setEnabled(False)
        self.run_match_button.setCheckable(True)           # ทำปุ่มเป็น toggle
        self.run_match_button.setEnabled(True)             # อนุญาตให้กดได้เลย (ไม่ต้องรอเปิดกล้อง/มีเทมเพลต)

        # ปรับขนาดและสไตล์ปุ่มให้ใหญ่ขึ้น
        btn_font = QFont()
        btn_font.setPointSize(11)  # ปรับขนาดตัวอักษร
        for b in (self.start_button, self.crop_button, self.open_template_button, self.run_match_button):
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
        controls_grid_layout.addWidget(self.open_template_button, 1, 0)
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
        title.setStyleSheet("font-size: 50px; font-weight: bold; color: #003366;")
        # counter label (top-right)
        self.counter_label = QLabel("OK:0  NG:0  NA:0  Total:0")
        self.counter_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.counter_label.setStyleSheet("font-size:50px; color:#003366; padding:0 10px 0 10px;")

        # เวลา (ย้ายไปอยู่บรรทัดบนของ counter และทำให้เล็กลง)
        self.time_label = QLabel("")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setStyleSheet("font-size:24px; color:#003366; padding:0 10px 0 10px;")

        # กล่องด้านขวารวมเวลา + ตัวนับ (แนวตั้ง)
        right_top_container = QWidget()
        right_top_v = QVBoxLayout(right_top_container)
        right_top_v.setContentsMargins(0, 0, 0, 0)
        right_top_v.setSpacing(0)
        right_top_v.addWidget(self.time_label, alignment=Qt.AlignRight)
        right_top_v.addWidget(self.counter_label, alignment=Qt.AlignRight)

        top_bar_layout = QHBoxLayout()
        top_bar_layout.addWidget(title)
        top_bar_layout.addStretch(1)
        # เดิมเคย addWidget(self.counter_label) / self.time_label แยกกัน -> เปลี่ยนเป็น container เดียว
        top_bar_layout.addWidget(right_top_container)
        overall_v_layout.addLayout(top_bar_layout)
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
        self.open_template_button.clicked.connect(self.open_template)
        self.run_match_button.toggled.connect(self.toggle_auto_match)
        # (Removed QTimer auto loop)
        self.auto_match_enabled = False
        self.matching_running = False
        self.destroyed.connect(self.cleanup)

        # load default template from machine_vision/temp.jpg (if exists)
        try:
            config = configparser.ConfigParser()
            # Correct search order: project root, sibling (legacy), local (last)
            cfg_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "config.ini"),   # machine_vision/config.ini
                os.path.join(os.path.dirname(__file__), "..", "config.ini"),         # (legacy) machine_vision/app/config.ini
                os.path.join(os.path.dirname(__file__), "config.ini")                # local fallback
            ]
            existing = [os.path.abspath(p) for p in cfg_paths if os.path.exists(os.path.abspath(p))]
            if existing:
                config.read(existing)
            template_name = "temp.jpg"
            if config.has_section("find_oring"):
                template_name = config.get("find_oring", "template_file", fallback=template_name)

            if os.path.isabs(template_name):
                default_template_path = template_name
            else:
                # resolve relative to project root (two levels up)
                default_template_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", template_name)
                )

            if os.path.exists(default_template_path):
                tpl = cv2.imread(default_template_path)
                if tpl is not None:
                    self.template_cv_image = tpl
                    self.image2.set_cv_image(self.template_cv_image, is_original=True)
                    self.open_template_button.setEnabled(True)
                    self.result_label.setText(f"Template loaded: {os.path.basename(default_template_path)}")
        except Exception:
            # keep silent to avoid UI crash; optionally add debug print if needed
            if os.environ.get("UBEND_DEBUG"):
                print("DEBUG: failed loading default template", flush=True)

        # โหลดยอดนับของวันนี้จากไฟล์ report (OK/NG/NA)
        self._load_today_counters()

        # เริ่มนาฬิกา
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_time)
        self._clock_timer.start(1000)
        self._update_time()

    # -------------------------------------------------------------
    # Reporting helpers: save datetime, input image, output image to CSV
    # Directory structure: report/YYYY/MM/
    # -------------------------------------------------------------
    def _get_report_month_dir(self) -> str:
        """Return absolute path to report/<YYYY>/<MM>, creating dirs as needed."""
        base_report_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "report")
        )
        now = datetime.now()
        year_dir = os.path.join(base_report_dir, now.strftime("%Y"))
        month_dir = os.path.join(year_dir, now.strftime("%m"))
        try:
            os.makedirs(month_dir, exist_ok=True)
        except Exception:
            # If creation fails for some reason, fall back to base dir
            month_dir = base_report_dir
            os.makedirs(month_dir, exist_ok=True)
        return month_dir

    def _save_report_entry(self, input_img: np.ndarray, output_img: np.ndarray, test_result: str = "") -> None:
        """Save input/output images + a CSV row: index,time,result,input_image,output_image
           test_result ต้องเป็นหนึ่งใน OK / NG / NA"""
        if input_img is None or output_img is None:
            return

        # Check config.ini report.save_images flag; if false, skip saving
        try:
            parser = configparser.ConfigParser()
            ini_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.ini"))
            if os.path.exists(ini_path):
                parser.read(ini_path)
                save_images = parser.getboolean('report', 'save_images', fallback=True)
            else:
                save_images = True
        except Exception:
            save_images = True

        if not save_images:
            # User disabled report image saving
            return

        month_dir = self._get_report_month_dir()
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        csv_path = os.path.join(month_dir, "report.csv")
        images_dir = os.path.join(month_dir, "image")
        try:
            os.makedirs(images_dir, exist_ok=True)
        except Exception:
            images_dir = month_dir

        write_header = not os.path.exists(csv_path)
        # Determine next index (ลำดับ) via csv.reader and handle old header migration
        index_val = 1
        if not write_header:
            try:
                with open(csv_path, mode="r", encoding="utf-8") as rf:
                    reader = csv.reader(rf)
                    rows = [r for r in reader if r]
                if rows:
                    header = rows[0]
                    data_rows = rows[1:]
                    # If old base64 header and no data, rewrite header to path-based
                    if (
                        header[:5] == ["index", "time", "result", "input_base64", "output_base64"]
                        and len(data_rows) == 0
                    ):
                        with open(csv_path, mode="w", newline="", encoding="utf-8") as wf:
                            writer = csv.writer(wf)
                            writer.writerow(["index", "time", "result", "input_image", "output_image"])
                        write_header = False
                        index_val = 1
                    else:
                        if len(rows) > 1:
                            last = rows[-1]
                            try:
                                index_val = int(last[0]) + 1
                            except Exception:
                                index_val = len(rows)  # header + (len(rows)-1) data
            except Exception:
                index_val = 1

        # Build filenames and save images
        stamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        in_name = f"{index_val:07d}_{stamp}_input.jpg"
        out_name = f"{index_val:07d}_{stamp}_output.jpg"
        in_path = os.path.join(images_dir, in_name)
        out_path = os.path.join(images_dir, out_name)
        try:
            cv2.imwrite(in_path, input_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            cv2.imwrite(out_path, output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except Exception:
            return
        try:
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["index", "time", "result", "input_image", "output_image"])
                writer.writerow([index_val, ts, test_result, in_path, out_path])
        except Exception:
            # Silently ignore CSV errors to avoid UI interruption
            pass

    def _load_today_counters(self):
        """อ่านไฟล์ report.csv ของเดือนปัจจุบัน แล้วรวมเฉพาะแถวของวันที่วันนี้ (result = OK/NG/NA)."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        try:
            month_dir = self._get_report_month_dir()
            csv_path = os.path.join(month_dir, "report.csv")
            if not os.path.exists(csv_path):
                return
            ok = ng = na = 0
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) < 3:
                        continue
                    time_field = row[1]
                    if not time_field.startswith(today_str):
                        continue
                    r = row[2].strip().upper()
                    if r == "OK":
                        ok += 1
                    elif r == "NG":
                        ng += 1
                    elif r == "NA":
                        na += 1
            self.count_ok = ok
            self.count_ng = ng
            self.count_na = na
            self.count_total = ok + ng + na
            self._refresh_counter_label()
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

        # Auto-run matching per frame if enabled
        if (
            self.auto_match_enabled
            and self.template_cv_image is not None
            and not self.matching_running
        ):
            self.run_match_on_current_frame()

    def toggle_crop_mode(self, checked):
        if not self.is_camera_on or self.cv_frame is None:
            self.crop_button.setChecked(False)
            return
        self.image1.set_cropping_mode(checked)
        self.result_label.setText("Crop mode ON" if checked else "Crop mode OFF")



    def save_template(self):
        if self.template_cv_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Template", "", "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)")
        if not path:
            return
        cv2.imwrite(path, self.template_cv_image)
        self.result_label.setText(f"Template saved: {os.path.basename(path)}")

    def open_template(self):
        """Open an existing template image from disk and load it as the current template."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Template Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if not path:
            return
        
        tpl = cv2.imread(path)
        if tpl is None:
            self.result_label.setText(f"Failed to load: {os.path.basename(path)}")
            return
        self.template_cv_image = tpl
        self.image2.set_cv_image(self.template_cv_image, is_original=True)
        # enable crop/save-ish UI as before (keep naming for minimal change)
        self._init_native_matcher()
        self.open_template_button.setEnabled(True)
        self.result_label.setText(f"Template loaded: {os.path.basename(path)}")

    def _init_native_matcher(self):
        """Initialize native matcher (matching2) with current template."""
        if tm_matching is None or self.template_cv_image is None:
            return
        try:
            gray = self.template_cv_image
            if gray.ndim == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            # create matcher only once (recreate when template changes)
            lib_path = tm_matching.resolve_lib_path()
            self._matcher = tm_matching.Matcher(
                lib_path,
                maxCount=1,
                scoreThreshold=0.55,
                iouThreshold=0.4,
                angle=360,
                minArea=64
            )
            self._matcher.set_template(gray)
            self._template_size = (gray.shape[1], gray.shape[0])
        except Exception as e:
            self._matcher = None
            self.result_label.setText(f"Matcher init error: {e}")
    
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
            input_for_log = self.cv_frame.copy()

            t0 = time.perf_counter()
            crop = self.rotation_matching2(self.cv_frame, self.template_cv_image)
            timings['rotation_matching'] = (time.perf_counter() - t0) * 1000.0

            if crop is None:
                self.result_label.setText("No match found")
                return

            self.cropped_cv_frame = crop

            t1 = time.perf_counter()
            self.image3.set_cv_image(crop, is_original=True)
            timings['set_image3'] = (time.perf_counter() - t1) * 1000.0

            # find_oring ตอนนี้คืน (final_img, classification)
            t2 = time.perf_counter()
            final_img, classification = self.find_oring(crop)
            timings['find_oring'] = (time.perf_counter() - t2) * 1000.0

            if final_img is None:
                self.result_label.setText("final_img not found")
                return

            t3 = time.perf_counter()
            self.image4.set_cv_image(final_img, is_original=True)
            timings['set_image4'] = (time.perf_counter() - t3) * 1000.0
            crop = None  # free memory
            # บันทึกเฉพาะ OK / NG / NA
            try:
                self._save_report_entry(input_for_log, final_img, classification)
            except Exception:
                pass

            total_ms = (time.perf_counter() - total_start) * 1000.0
            # self.result_label.setText(f"{classification} | Total {total_ms:.1f}ms")
        finally:
            self.matching_running = False
            # self.cv_frame = None

    def toggle_auto_match(self, checked: bool):
        """เปิด/ปิดโหมดจับคู่ต่อเฟรม; ถ้ายังไม่เปิดกล้องจะเปิดให้เอง"""
        if checked:
            # ถ้ายังไม่เปิดกล้อง ให้เปิดอัตโนมัติ
            if not self.is_camera_on:
                self.toggle_camera()
                # รอให้ได้อย่างน้อย 1 เฟรม (แบบง่าย ๆ ไม่ block นาน)
            if self.template_cv_image is None:
                self.result_label.setText("No template (ให้ CROP หรือโหลดก่อน)")
                self.run_match_button.setChecked(False)
                return
            self.auto_match_enabled = True
            self.run_match_button.setText("STOP")
            self.result_label.setText("Auto-match ON")
        else:
            self.auto_match_enabled = False
            self.run_match_button.setText("RUN")
            self.result_label.setText("Auto-match OFF")

    def rotation_matching(self, target_image, template):

        start_t = time.time()
        TIMEOUT = 12.0

        if template is None or template.size == 0:
            self.result_label.setText("No template.")
            return None

        src = target_image.copy()
        target_result = target_image.copy()        # <-- ensure defined
        try:
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.result_label.setText(f"Color convert error: {e}")
            return None

        th, tw = template_gray.shape

        try:
            matches = app.matching.match_template_rotated_coarse_to_fine(
                target_gray, template_gray,
                coarse_step=6,
                fine_step=0.1,
                coarse_range=(-180, 180),
                fine_window=0.1,
                candidates_per_angle=3,
                min_score=0.5,
                scale=0.08,
                iou_thresh=0.2
            )
        except Exception as e:
            self.result_label.setText(f"Match error: {e}")
            return None

        crop = None
        for i, match in enumerate(matches):
            pos = match["position"]
            angle = match["angle"]
            score = match["score"]

            # angle tweaks
            if -80.0 < angle < -75.0:
                angle -= 6
                pos = (pos[0], pos[1] + 20)
            if -133.0 < angle < -130.0:
                pos = (pos[0] + 40, pos[1] + 50)
            if 105.0 < angle < 110.0:
                angle -= 10
                pos = (pos[0] + 20, pos[1] - 20)

            # draw (safe to overwrite each loop; we break after first valid crop)
            target_result = app.matching.draw_rotated_match(
                src, pos, tw, th, angle,
                label=f"#{i} {angle:.1f}° {score:.2f}"
            )

            rotated_src, rot_mat = app.matching.rotate_image(target_image, angle, border_value=255)
            center_in_rot = app.matching.apply_affine_transform(pos, rot_mat)
            cx_r, cy_r = map(lambda v: int(round(v)), center_in_rot)

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
                break  # first valid match

        if crop is None:
            self.result_label.setText("No match found")
            self.image3.set_cv_image(None)
        else:
            self.image3.set_cv_image(crop, is_original=True)

        self.image1.set_cv_image(target_result, is_original=True)
        QApplication.processEvents()
        return crop

    def rotation_matching2(self, target_image, template):
        """Use native matcher (matching2). Returns rectified crop or None."""
        if self._matcher is None:
            self._init_native_matcher()
            if self._matcher is None:
                self.result_label.setText("Matcher not ready")
                return None
        if target_image is None or target_image.size == 0:
            self.result_label.setText("No target frame")
            return None
        try:
            # Run native match (internally converts to gray)
            match_count = self._matcher.match(target_image)
        except Exception as e:
            self.result_label.setText(f"Match call error: {e}")
            return None
        if match_count <= 0:
            self.result_label.setText("No match")
            return None

        res = self._matcher.results[0]
        if res.score <= 0:
            self.result_label.setText("Low score")
            return None

        # Corner order from wrapper: leftTop, leftBottom, rightBottom, rightTop
        corners = np.array([
            [res.leftTopX, res.leftTopY],
            [res.leftBottomX, res.leftBottomY],
            [res.rightBottomX, res.rightBottomY],
            [res.rightTopX, res.rightTopY]
        ], dtype=np.float32)

        draw_img = target_image.copy()
        pts_int = corners.astype(np.int32).reshape(-1,1,2)
        cv2.polylines(draw_img, [pts_int], True, (0,255,0), 2)
        cv2.putText(draw_img, f"{res.angle:.1f}deg {res.score:.2f}",
                    (int(res.centerX)+5, int(res.centerY)+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        self.image1.set_cv_image(draw_img, is_original=True)

        # Warp to template size (rectified crop)
        if self._template_size is None:
            self.result_label.setText("No template size")
            return None
        tw, th = self._template_size
        dst_quad = np.array([
            [0,0],
            [0,th-1],
            [tw-1,th-1],
            [tw-1,0]
        ], dtype=np.float32)

        try:
            M = cv2.getPerspectiveTransform(corners, dst_quad)
            warped = cv2.warpPerspective(target_image, M, (tw, th),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255,255,255))
        except Exception as e:
            self.result_label.setText(f"Warp error: {e}")
            return None

        self.image3.set_cv_image(warped, is_original=True)
        QApplication.processEvents()
        return warped

    def find_oring(self, image: np.ndarray,
                   large_area: int = 5000,
                   dif_zone: int = 0,
                   h_oring: int = 65,
                   w_oring: int = 110):
        cfg = self._read_find_oring_config()
        def _iget(key, default):
            try:
                return int(cfg.get(key, default))
            except Exception:
                return default
        def _fget(key, default):
            try:
                return float(cfg.get(key, default))
            except Exception:
                return default
        
        large_area = _iget("large_area", large_area)
        dif_zone = _iget("dif_zone", dif_zone)
        h_oring = _iget("h_oring", h_oring)
        w_oring = _iget("w_oring", w_oring)
        blur_kernel = _iget("blur_kernel", 15)
        canny_thresh1 = _iget("canny_thresh1", 50)
        canny_thresh2 = _iget("canny_thresh2", 90)
        morph_kernel = _iget("morph_kernel", 15)
        rect_height_ratio = _fget("rect_height_ratio", 0.5)
        object_thresh = _iget("object_thresh", 0)
        object_maxval = _iget("object_maxval", 100)   # จาก config.ini ตัวอย่างเป็น 100
        hole_thresh = _iget("hole_thresh", 30)
        hole_maxval = _iget("hole_maxval", 255)

        """Process oring detection. Return (processed_image, classification) where classification in OK/NG/NA."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return (np.zeros((400, 600, 3), dtype=np.uint8), "NA")
        
        if blur_kernel % 2 == 0:
            blur_kernel = max(3, blur_kernel - 1)
        if morph_kernel <= 0:
            morph_kernel = 15

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
        channel = s
        s_increased = cv2.add(channel, 8)
        
        _, object_mask = cv2.threshold(s_increased, object_thresh, object_maxval, cv2.THRESH_BINARY)
        _, hole_mask = cv2.threshold(s_increased, hole_thresh, hole_maxval, cv2.THRESH_BINARY_INV)
        inpaint_mask = cv2.bitwise_and(object_mask, hole_mask)
        try:
            _ = cv2.inpaint(s_increased, inpaint_mask, 1, cv2.INPAINT_NS)
        except Exception:
            pass

        blurred = cv2.GaussianBlur(inpaint_mask, (blur_kernel, blur_kernel), 1)
        _, smoothed = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(smoothed, canny_thresh1, canny_thresh2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, morph_kernel//2)|1, max(3, morph_kernel//2)|1))

        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
        work = closed
        contours2, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("debug_edges", work)
        cv2.imshow("debug_smoothed", smoothed)

        large_contours = [c for c in contours2 if cv2.contourArea(c) >= large_area]

        # ปริ้นแสดงพื้นที่ area contour ที่หาได้ (debug)
        
        for i, c in enumerate(contours2):
            print(f"find_oring: contour[{i}] area=", cv2.contourArea(c))
            self.result_label.setText(f"find_oring: contour[{i}] area={cv2.contourArea(c)}")
        # Fallback: ถ้าไม่มี large_contours ให้ลองลดเกณฑ์และลอง morphology เพื่อช่วยเชื่อมรูปร่าง
        # if not large_contours and contours2:
        #     # ใช้ top few contours แทน (ลดเกณฑ์)
        #     small_sorted = sorted(contours2, key=cv2.contourArea, reverse=True)[:4]
        #     large_contours = small_sorted

        # if not large_contours:
        #     # morphology fallback
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        #     closed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        #     edges2 = cv2.Canny(closed, max(10, canny_thresh1 - 10), min(255, canny_thresh2 + 10))
        #     contours2b, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     large_contours = [c for c in contours2b if cv2.contourArea(c) >= large_area]

        #     try:
        #         if os.environ.get("UBEND_DEBUG"):
        #             print("find_oring: fallback contours after morph=", len(contours2b), "large_after=", len(large_contours))
        #             cv2.imwrite("/tmp/oring_smoothed.png", smoothed)
        #             cv2.imwrite("/tmp/oring_edges.png", edges)
        #             cv2.imwrite("/tmp/oring_edges2.png", edges2)
        #     except Exception:
        #         pass

        overlay = img.copy()
        # if large_contours:
        #     cv2.drawContours(overlay, large_contours, -1, (0, 200, 0), 1)
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
            fill_color = (0, 255, 0) if (diff_val > h_oring and rect_width > w_oring) else (255, 0, 0)
            text_result = "OK" if (diff_val > h_oring and rect_width > w_oring) else "NG"
            zone_status[z] = text_result
            overlay2 = p.copy()
            cv2.rectangle(overlay2, (x_min, top_y), (x_max, bottom_y), fill_color, -1)
            p = cv2.addWeighted(overlay2, 0.4, p, 0.6, 0)
            cv2.rectangle(p, (x_min, top_y), (x_max, bottom_y), (255, 0, 0), 1)

            debug_text = f"Z{z} d:{diff_val} w:{rect_width}"
            cv2.putText(p, debug_text, (x_min, max(0, top_y - 8)), font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            # if os.environ.get("UBEND_DEBUG"):
                # print(f"[find_oring] zone={z} diff_val={diff_val} rect_width={rect_width} h_oring={h_oring} w_oring={w_oring}", flush=True)
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
        # ---- update counters ----
        data = None
        if s1 == "OK" and s2 == "OK":
            classification = "OK"
            data = "OK"
        elif s1 == "NG" or s2 == "NG":
            classification = "NG"
            data = "NG"
        else:
            classification = "NA"
            data = "NA"
        self._increment_counters(classification)
        try:
            self.classification_emitted.emit(classification)   # <-- emit result
            self._send_classification_udp(data)  # <-- send via UDP
        except Exception:
            pass
        return p, classification

    def cleanup(self):
        if self.auto_match_enabled:
            self.auto_match_enabled = False
            self.run_match_button.setChecked(False)
            self.run_match_button.setText("RUN")
        if self.camera_worker and self.camera_worker.isRunning():
             self.camera_worker.stop()
             self.camera_worker.wait()
             self.camera_worker = None

    # ---- counter helpers ----
    def _increment_counters(self, cls: str):
        self.count_total += 1
        if cls == "OK":
            self.count_ok += 1
        elif cls == "NG":
            self.count_ng += 1
        else:
            self.count_na += 1
        self._refresh_counter_label()

    def _refresh_counter_label(self):
        if hasattr(self, "counter_label"):
            self.counter_label.setText(
                f"OK:{self.count_ok}  NG:{self.count_ng}  NA:{self.count_na}  Total:{self.count_total}"
            )

    def _update_time(self):
        """Update the time label to show current system time."""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            if hasattr(self, "time_label"):
                self.time_label.setText(current_time)
        except Exception:
            pass

    # def finish_crop(self):
    #     rect = self.image1.get_crop_rect_on_original()
    #     self.image1.set_cropping_mode(False)
    #     self.crop_button.setChecked(False)
    #     if rect is None:
    #         self.result_label.setText("Invalid crop")
    #         return
    #     self.template_cv_image = self.image1.original_cv_image[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()].copy()
    #     # update template preview and force repaint/layout so it's visible inside the QGroupBox
    #     self.image2.set_cv_image(self.template_cv_image, is_original=True)
    #     try:
    #         self.image2.update()
    #         self.image2.repaint()
    #         if hasattr(self, "image2_group"):
    #             self.image2_group.updateGeometry()
    #             self.image2_group.repaint()
    #         QApplication.processEvents()
    #     except Exception:
    #         pass
    #     self.open_template_button.setEnabled(True)
    #     self.result_label.setText("Template created")
    #     # NEW: switch crop button to SAVE MATCHING mode
    #     self._switch_crop_button_to_save()

    def finish_crop(self):
        rect = self.image1.get_crop_rect_on_original()
        self.image1.set_cropping_mode(False)
        self.crop_button.setChecked(False)
        if rect is None:
            self.result_label.setText("Invalid crop")
            return
        self.template_cv_image = self.image1.original_cv_image[
            rect.y():rect.y()+rect.height(),
            rect.x():rect.x()+rect.width()
        ].copy()
        self.image2.set_cv_image(self.template_cv_image, is_original=True)
        # INIT matcher with new template
        self._init_native_matcher()
        try:
            self.image2.update(); self.image2.repaint()
            if hasattr(self, "image2_group"):
                self.image2_group.updateGeometry()
                self.image2_group.repaint()
            QApplication.processEvents()
        except Exception:
            pass
        self.open_template_button.setEnabled(True)
        self.result_label.setText("Template created")
        self._switch_crop_button_to_save()

    def _switch_crop_button_to_save(self):
        """Change the crop button into a SAVE MATCHING button after finishing a crop."""
        try:
            self.crop_button.toggled.disconnect(self.toggle_crop_mode)
        except Exception:
            pass
        try:
            self.crop_button.clicked.disconnect(self._save_matching_action)
        except Exception:
            pass
        self.crop_button.setCheckable(False)
        self.crop_button.setText("SAVE MATCHING")
        self.crop_button.clicked.connect(self._save_matching_action)

    def _restore_crop_button(self):
        """Restore button back to CROP functionality."""
        try:
            self.crop_button.clicked.disconnect(self._save_matching_action)
        except Exception:
            pass
        try:
            self.crop_button.toggled.disconnect(self.toggle_crop_mode)
        except Exception:
            pass
        self.crop_button.setText("MATCHING CROP")
        self.crop_button.setCheckable(True)
        self.crop_button.toggled.connect(self.toggle_crop_mode)

    def _save_matching_action(self):
        """Handle save matching click; only revert button if save succeeds."""
        if self.template_cv_image is None:
            self.result_label.setText("No template to save")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Matching Template",
            "",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)"
        )
        if not path:
            # user canceled; keep button as SAVE MATCHING
            return
        try:
            cv2.imwrite(path, self.template_cv_image)
            self.result_label.setText(f"Template saved: {os.path.basename(path)}")
        except Exception as e:
            self.result_label.setText(f"Save failed: {e}")
            return
        # revert button back to crop mode
        self._restore_crop_button()

    def _read_find_oring_config(self):
        """อ่านค่า section [find_oring] จาก config.ini (ค้นหลาย path) ทุกครั้ง"""
        cfg = {}
        try:
            base_dir = os.path.dirname(__file__)
            candidate_paths = [
                os.path.abspath(os.path.join(base_dir, "..", "..", "config.ini")),   # project root: machine_vision/config.ini
                os.path.abspath(os.path.join(base_dir, "..", "config.ini")),        # fallback: machine_vision/app/config.ini
                os.path.abspath(os.path.join(base_dir, "config.ini")),              # fallback: machine_vision/app/ui/config.ini
            ]
            existing = [p for p in candidate_paths if os.path.exists(p)]
            if not existing:
                if os.environ.get("UBEND_DEBUG"):
                    print("DEBUG _read_find_oring_config: not found in", candidate_paths, flush=True)
                return cfg
            parser = configparser.ConfigParser()
            parser.read(existing, encoding="utf-8")
            if parser.has_section("find_oring"):
                for k, v in parser["find_oring"].items():
                    cfg[k.strip()] = v.strip()
            else:
                if os.environ.get("UBEND_DEBUG"):
                    print("DEBUG _read_find_oring_config: section [find_oring] missing", flush=True)
        except Exception as e:
            if os.environ.get("UBEND_DEBUG"):
                print(f"DEBUG _read_find_oring_config error: {e}", flush=True)
        return cfg

    def _load_tcp_config(self):
        """Load TCP sender config from any found config.ini (shared search logic)."""
        try:
            base_dir = os.path.dirname(__file__)
            candidate_paths = [
                os.path.abspath(os.path.join(base_dir, "..", "..", "config.ini")),
                os.path.abspath(os.path.join(base_dir, "..", "config.ini")),
                os.path.abspath(os.path.join(base_dir, "config.ini")),
            ]
            parser = configparser.ConfigParser()
            existing = [p for p in candidate_paths if os.path.exists(p)]
            if not existing:
                return
            parser.read(existing, encoding="utf-8")
            if parser.has_section("tcp"):
                self.tcp_enabled = parser.getboolean("tcp", "enable", fallback=False)
                self.tcp_host = parser.get("tcp", "host", fallback=self.tcp_host)
                self.tcp_port = parser.getint("tcp", "port", fallback=self.tcp_port)
                self.tcp_mode = parser.get("tcp", "mode", fallback=self.tcp_mode).lower()
        except Exception:
            pass

    def _send_classification_udp(self, classification: str):
        """Send classification via UDP (fire & forget)."""
        if not self.tcp_enabled:
            return
        classification = (classification or "NA").upper()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.tcp_mode == "json":
            import json
            payload = {
                "type": "classification",
                "value": classification,
                "time": timestamp,
                "counters": {
                    "ok": self.count_ok,
                    "ng": self.count_ng,
                    "na": self.count_na,
                    "total": self.count_total
                }
            }
            msg = json.dumps(payload) + "\n"
        else:
            msg = f"CLASSIFICATION,{classification},{timestamp},OK={self.count_ok},NG={self.count_ng},NA={self.count_na},TOTAL={self.count_total}\n"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.sendto(msg.encode('utf-8'), (self.tcp_host, self.tcp_port))
        except Exception:
            if os.environ.get("UBEND_DEBUG"):
                print(f"[UDP] send failed to {self.tcp_host}:{self.tcp_port}", flush=True)
        finally:
            try:
                sock.close()
            except Exception:
                pass
    