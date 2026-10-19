from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QFormLayout,
    QGroupBox, QGridLayout, QSizePolicy, QSpinBox, QHBoxLayout, QLayout, QCheckBox,
    QComboBox, QDoubleSpinBox, QScrollArea, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer, QSize
import configparser
import os
from collections import deque
import cv2
import numpy as np
from PySide6.QtGui import QIntValidator, QDoubleValidator, QFont, QImage, QPixmap
from app import ini_store
from app.oring_settings import OringSettings, load_settings
from app.oring_v2 import find_oring_v2
# --- new imports for serial listing ---
import glob
try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:
    serial = None
    list_ports = None

class ScaledImageLabel(QLabel):
    """Shows a picture scaled to the label (keeps the aspect ratio) and re-scales it when the label is resized.
    The size hints are fixed, so the picture never influences the layout (no resize feedback loop)."""
    def __init__(self, text=""):
        super().__init__(text)
        self._src = QPixmap()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)

    def sizeHint(self):
        return QSize(640, 420)

    def minimumSizeHint(self):
        return QSize(320, 240)

    def set_source(self, pix: QPixmap):
        self._src = pix
        self._fit()

    def _fit(self):
        if not self._src.isNull() and self.width() > 1 and self.height() > 1:
            self.setPixmap(self._src.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class SettingPage(QWidget):
    oring_settings_changed = Signal(object)     # OringSettings, after "Save Settings" (HomePage.apply_oring_settings)

    def __init__(self, data_count: int = 2,
                 plc_addr_start: int = 0, plc_addr_step: int = 2,
                 data_default: str = "NA", row_spacing: int = 22):
        super().__init__()
        # use same config.ini path as other modules (machine_vision/config.ini)
        self.config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.ini"))
        self.config = configparser.ConfigParser()
        self.load_config()
        self.oring_settings = load_settings()        # OK/NG judgement ([oring_v2] of the same file)
        self._recent = deque(maxlen=30)              # latest parts from HOME: dict(crop, label)
        self._leg_stats = deque(maxlen=200)          # latest measured legs: dict(dist, bump, width)
        self._preview_pos = 0                        # 0 = newest part

        # the page can be taller than the screen: everything lives in a scroll area
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 4, 4)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel("SETTINGS")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-size: 50px; font-weight: bold; color: #003366;")
        layout.addWidget(title)
        
        # OK / NG judgement (new detector) + live preview
        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        top_row.addWidget(self._build_judgement_group(), 0, Qt.AlignTop)
        top_row.addWidget(self._build_preview_group(), 1)          # stretches to the height of the judgement group
        layout.addLayout(top_row)

        # Group box for the legacy O-ring settings (only used by engine = legacy)
        oring_group = QGroupBox("setting oring (legacy)")
        self.oring_group = oring_group
        oring_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        oring_group.setFixedWidth(420)
        
        form_layout = QFormLayout(oring_group)

        # use safe fallback values from config
        self.large_area_input = QLineEdit(str(self.config.getint('find_oring', 'large_area', fallback=5000)))
        self.dif_zone_input = QLineEdit(str(self.config.getint('find_oring', 'dif_zone', fallback=0)))
        self.h_oring_input = QLineEdit(str(self.config.getint('find_oring', 'h_oring', fallback=65)))
        self.w_oring_input = QLineEdit(str(self.config.getint('find_oring', 'w_oring', fallback=110)))
        # compact width + integer validator + right alignment
        int_validator = QIntValidator(0, 999999, self)
        for le in (self.large_area_input, self.dif_zone_input, self.h_oring_input, self.w_oring_input):
            le.setValidator(int_validator)
            le.setFixedWidth(90)
            le.setAlignment(Qt.AlignRight)
 
         # image-processing params
        self.blur_kernel_input = QLineEdit(str(self.config.getint('find_oring', 'blur_kernel', fallback=15)))
        self.canny1_input = QLineEdit(str(self.config.getint('find_oring', 'canny_thresh1', fallback=50)))
        self.canny2_input = QLineEdit(str(self.config.getint('find_oring', 'canny_thresh2', fallback=90)))
        self.morph_kernel_input = QLineEdit(str(self.config.getint('find_oring', 'morph_kernel', fallback=15)))
        self.rect_ratio_input = QLineEdit(str(self.config.get('find_oring', 'rect_height_ratio', fallback='0.5')))
        # validators and sizing for image-processing inputs
        for le in (self.blur_kernel_input, self.canny1_input, self.canny2_input, self.morph_kernel_input):
            le.setValidator(QIntValidator(0, 9999, self))
            le.setFixedWidth(80)
            le.setAlignment(Qt.AlignRight)
        # rect ratio: float between 0.0 and 10.0 with 2 decimals
        dbl_val = QDoubleValidator(0.0, 10.0, 2, self)
        dbl_val.setNotation(QDoubleValidator.StandardNotation)
        self.rect_ratio_input.setValidator(dbl_val)
        self.rect_ratio_input.setFixedWidth(80)
        self.rect_ratio_input.setAlignment(Qt.AlignRight)

        form_layout.addRow(QLabel("Large Area:"), self.large_area_input)
        form_layout.addRow(QLabel("Difference Zone:"), self.dif_zone_input)
        form_layout.addRow(QLabel("Height of Oring:"), self.h_oring_input)
        form_layout.addRow(QLabel("Width of Oring:"), self.w_oring_input)
        # add new rows
        form_layout.addRow(QLabel("Blur kernel (odd):"), self.blur_kernel_input)
        form_layout.addRow(QLabel("Canny thresh1:"), self.canny1_input)
        form_layout.addRow(QLabel("Canny thresh2:"), self.canny2_input)
        form_layout.addRow(QLabel("Morph kernel:"), self.morph_kernel_input)
        form_layout.addRow(QLabel("Rect height ratio:"), self.rect_ratio_input)

        layout.addWidget(oring_group)
        oring_group.setVisible(self.oring_settings.engine == "legacy")

        # --- Report group: choice to save images in report ---
        report_group = QGroupBox("REPORT")
        report_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        report_group.setFixedWidth(420)
        report_layout = QHBoxLayout(report_group)
        self.save_reports_checkbox = QCheckBox("Save images for report")
        # read boolean from config file (fallback True)
        save_images_default = self.config.getboolean('report', 'save_images', fallback=True)
        self.save_reports_checkbox.setChecked(bool(save_images_default))
        report_layout.addWidget(self.save_reports_checkbox)
        report_layout.addStretch()
        layout.addWidget(report_group)
        # --- end report group ---

        # --- SERIAL / MODBUS group ---
        serial_group = QGroupBox("SERIAL / MODBUS")
        serial_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        serial_group.setFixedWidth(420)
        s_form = QFormLayout(serial_group)

        # Port selection + refresh
        port_row = QHBoxLayout()
        self.serial_port_combo = QComboBox()
        self.serial_port_combo.setMinimumWidth(180)
        self.serial_refresh_btn = QPushButton("Refresh")
        self.serial_refresh_btn.setFixedWidth(70)
        self.serial_refresh_btn.clicked.connect(self.populate_serial_ports)
        port_row.addWidget(self.serial_port_combo)
        port_row.addWidget(self.serial_refresh_btn)
        port_row.addStretch()
        port_row_w = QWidget()
        port_row_w.setLayout(port_row)
        s_form.addRow(QLabel("Port:"), port_row_w)

        # Baudrate
        self.serial_baud_combo = QComboBox()
        self.serial_baud_combo.addItems(["9600","19200","38400","57600","115200"])
        s_form.addRow(QLabel("Baud:"), self.serial_baud_combo)

        # Parity
        self.serial_parity_combo = QComboBox()
        self.serial_parity_combo.addItems(["NONE","EVEN","ODD","MARK","SPACE"])
        s_form.addRow(QLabel("Parity:"), self.serial_parity_combo)

        # Stop bits
        self.serial_stopbits_combo = QComboBox()
        self.serial_stopbits_combo.addItems(["1","1.5","2"])
        s_form.addRow(QLabel("Stop Bits:"), self.serial_stopbits_combo)

        # Data bits
        self.serial_bytesize_combo = QComboBox()
        self.serial_bytesize_combo.addItems(["7","8"])
        s_form.addRow(QLabel("Data Bits:"), self.serial_bytesize_combo)

        # Slave ID
        self.modbus_slave_spin = QSpinBox()
        self.modbus_slave_spin.setRange(1,247)
        self.modbus_slave_spin.setValue(1)
        self.modbus_slave_spin.setFixedWidth(80)
        s_form.addRow(QLabel("Slave ID:"), self.modbus_slave_spin)

        # Test button
        self.serial_test_btn = QPushButton("Test Connect")
        self.serial_test_btn.setFixedWidth(120)
        self.serial_test_btn.clicked.connect(self.test_serial_connection)
        s_form.addRow(QLabel(""), self.serial_test_btn)

        layout.addWidget(serial_group)
        # --- end SERIAL / MODBUS group ---

        # --- I/O Page content integration ---
        self.data_count = max(1, data_count)
        self.plc_addr_start = plc_addr_start
        self.plc_addr_step = plc_addr_step
        self.data_default = (data_default or "NA").upper()
        self.row_spacing = max(4, int(row_spacing))

        self.states = {}
        self.widgets = {}

        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(8)
        lbl_count = QLabel("DATA count:")
        lbl_count.setFixedWidth(70)
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 24)
        self.spin_count.setValue(self.data_count)
        self.spin_count.setFixedWidth(60)
        self.spin_count.valueChanged.connect(self.set_data_count)
        ctrl_layout.addWidget(lbl_count)
        ctrl_layout.addWidget(self.spin_count)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        self.group = QGroupBox("I/O SETTING")
        self.group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.group.setFixedWidth(420)
        self.group_layout = QGridLayout()
        self.group_layout.setContentsMargins(6, 6, 6, 6)
        self.group_layout.setHorizontalSpacing(12)
        self.group_layout.setVerticalSpacing(max(4, self.row_spacing // 2))
        self.group_layout.setSizeConstraint(QLayout.SetMinimumSize)
        self.group.setLayout(self.group_layout)
        layout.addWidget(self.group, 0, Qt.AlignTop | Qt.AlignLeft)

        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(9)
        lbl_machine = QLabel("MACHINE VISION")
        lbl_machine.setFont(header_font)
        lbl_plc = QLabel("PLC OUTPUT")
        lbl_plc.setFont(header_font)
        self.group_layout.addWidget(lbl_machine, 0, 0, 1, 2, Qt.AlignLeft)
        self.group_layout.addWidget(lbl_plc, 0, 3, 1, 1, Qt.AlignLeft)

        for i in range(1, self.data_count + 1):
            key = f"data{i}"
            self._create_row(i, key)
        self._adjust_group_height()
        # --- End of I/O Page integration ---

        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_config)
        self.save_button.setFixedWidth(420)
        layout.addWidget(self.save_button)
        layout.addStretch()

        # ensure at least first two rows exist (DATA1 / DATA2) for mapping OK / NG
        if self.data_count < 2:
            self.set_data_count(2)
        # apply serial config after widgets created
        self._apply_serial_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.config['find_oring'] = {
                'large_area': '5000',
                'dif_zone': '0',
                'h_oring': '65',
                'w_oring': '110',
                'blur_kernel': '15',
                'canny_thresh1': '50',
                'canny_thresh2': '90',
                'morph_kernel': '15',
                'rect_height_ratio': '0.5'
            }
            # default report section
            self.config['report'] = {
                'save_images': 'true'
            }
        if not self.config.has_section('serial'):
            self.config['serial'] = {
                'port': '',
                'baudrate': '9600',
                'parity': 'NONE',
                'stopbits': '1',
                'bytesize': '8',
                'slave_id': '1'
            }

    def save_config(self):
        # start from the file as it is NOW: HomePage (selected model) and others write to it after this page was created
        self.config = ini_store.read_ini(self.config_file)
        # ensure section exists
        if not self.config.has_section('find_oring'):
            self.config.add_section('find_oring')
        self.config['find_oring']['large_area'] = self.large_area_input.text()
        self.config['find_oring']['dif_zone'] = self.dif_zone_input.text()
        self.config['find_oring']['h_oring'] = self.h_oring_input.text()
        self.config['find_oring']['w_oring'] = self.w_oring_input.text()
        # save new params
        self.config['find_oring']['blur_kernel'] = self.blur_kernel_input.text()
        self.config['find_oring']['canny_thresh1'] = self.canny1_input.text()
        self.config['find_oring']['canny_thresh2'] = self.canny2_input.text()
        self.config['find_oring']['morph_kernel'] = self.morph_kernel_input.text()
        self.config['find_oring']['rect_height_ratio'] = self.rect_ratio_input.text()

        # report section
        if not self.config.has_section('report'):
            self.config.add_section('report')
        self.config['report']['save_images'] = 'true' if self.save_reports_checkbox.isChecked() else 'false'

        # serial section
        if not self.config.has_section('serial'):
            self.config.add_section('serial')
        self.config['serial']['port'] = self.serial_port_combo.currentText().strip()
        self.config['serial']['baudrate'] = self.serial_baud_combo.currentText()
        self.config['serial']['parity'] = self.serial_parity_combo.currentText()
        self.config['serial']['stopbits'] = self.serial_stopbits_combo.currentText()
        self.config['serial']['bytesize'] = self.serial_bytesize_combo.currentText()
        self.config['serial']['slave_id'] = str(self.modbus_slave_spin.value())

        # O-ring judgement ([oring_v2])
        settings = self._collect_oring_settings()
        if not self.config.has_section('oring_v2'):
            self.config.add_section('oring_v2')
        for k, v in settings.to_section().items():
            self.config['oring_v2'][k] = str(v)

        try:
            ini_store.write_ini(self.config, self.config_file, backup=True)
        except OSError as e:
            QMessageBox.critical(self, "Save Settings", f"บันทึกไม่สำเร็จ: {e}")
            return
        self.oring_settings = settings
        self.oring_settings_changed.emit(settings)
        self.save_button.setText("Saved ✓")
        QTimer.singleShot(1500, lambda: self.save_button.setText("Save Settings"))

    # ------------------------------------------------------------------
    # O-RING JUDGEMENT (OK / NG settings) + live preview of the latest parts
    # ------------------------------------------------------------------
    def _build_judgement_group(self) -> QGroupBox:
        st = self.oring_settings
        g = QGroupBox("O-RING JUDGEMENT  (เกณฑ์ตัดสิน OK / NG)")
        g.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        g.setFixedWidth(470)
        form = QFormLayout(g)
        form.setLabelAlignment(Qt.AlignLeft)

        self.dist_nominal_spin = QSpinBox()
        self.dist_nominal_spin.setRange(0, 300)
        self.dist_nominal_spin.setSuffix(" px")
        self.dist_nominal_spin.setValue(st.dist_nominal_px)
        self.dist_tol_spin = QSpinBox()
        self.dist_tol_spin.setRange(0, 150)
        self.dist_tol_spin.setPrefix("± ")
        self.dist_tol_spin.setSuffix(" px")
        self.dist_tol_spin.setValue(st.dist_tol_px)
        for w in (self.dist_nominal_spin, self.dist_tol_spin):
            w.setFixedWidth(100)
        dist_row = QHBoxLayout()
        dist_row.addWidget(self.dist_nominal_spin)
        dist_row.addWidget(self.dist_tol_spin)
        dist_row.addStretch()
        dist_w = QWidget()
        dist_w.setLayout(dist_row)
        dist_w.setToolTip("ระยะจากแหวนถึงปลายท่อ: ค่ากลาง ± ความคลาดเคลื่อน\nแหวนอยู่นอกช่วงนี้ = NG")
        form.addRow(QLabel("ตำแหน่งแหวน\n(ระยะจากปลายท่อ):"), dist_w)
        self.dist_range_label = QLabel()
        self.dist_range_label.setStyleSheet("color:#003366; font-weight:bold;")
        form.addRow(QLabel(""), self.dist_range_label)

        self.min_bump_spin = QDoubleSpinBox()
        self.min_bump_spin.setRange(0.0, 50.0)
        self.min_bump_spin.setSingleStep(0.5)
        self.min_bump_spin.setDecimals(1)
        self.min_bump_spin.setSuffix(" px")
        self.min_bump_spin.setValue(st.min_bump_px)
        self.min_bump_spin.setFixedWidth(100)
        self.min_bump_spin.setToolTip("แหวนต้องยื่นออกจากผิวท่ออย่างน้อยเท่านี้ ถึงจะนับว่ามีแหวน\n"
                                      "ค่าสูง = เข้มงวดขึ้น (NG มากขึ้น)  ค่าต่ำ = ผ่อนปรน\n"
                                      "ข้อมูลจริง: มีแหวน 10-24 px, ไม่มีแหวน 0-3 px")
        form.addRow(QLabel("แหวนต้องนูนอย่างน้อย:"), self.min_bump_spin)

        self.min_width_spin = QSpinBox()
        self.min_width_spin.setRange(0, 400)
        self.min_width_spin.setSuffix(" px")
        self.min_width_spin.setSpecialValueText("ไม่ตรวจ")
        self.min_width_spin.setValue(st.min_width_px)
        self.min_width_spin.setFixedWidth(100)
        self.min_width_spin.setToolTip("ความกว้างของแหวนขั้นต่ำ (0 = ไม่ตรวจ)")
        form.addRow(QLabel("ความกว้างแหวนขั้นต่ำ:"), self.min_width_spin)

        self.na_combo = QComboBox()
        for label, value in (("NG  (ถือว่าไม่ผ่าน)", "NG"), ("NA  (ส่งตามจริง)", "NA"), ("OK  (ถือว่าผ่าน)", "OK")):
            self.na_combo.addItem(label, value)
        self.na_combo.setCurrentIndex(max(0, self.na_combo.findData(st.na_send_as)))
        self.na_combo.setToolTip("ชิ้นงานที่ประเมินไม่ได้ (NA) เช่น ซ้อนกัน/เอียงมาก/ถูกตัดขอบ\nจะส่งไป PLC / UDP เป็นค่านี้ (หน้าจอและรายงานยังแสดงเป็น NA)")
        form.addRow(QLabel("ผล NA ส่งไป PLC/UDP เป็น:"), self.na_combo)

        self.show_measure_check = QCheckBox("แสดงตัวเลขที่วัดบนภาพผลลัพธ์")
        self.show_measure_check.setChecked(st.show_measure)
        form.addRow(QLabel(""), self.show_measure_check)

        self.recent_label = QLabel("ค่าที่วัดล่าสุด: -")
        self.recent_label.setWordWrap(True)
        self.recent_label.setStyleSheet("color:#444;")
        form.addRow(self.recent_label)

        reset_btn = QPushButton("คืนค่าเริ่มต้น")
        reset_btn.setFixedWidth(130)
        reset_btn.clicked.connect(self._reset_judgement_defaults)
        form.addRow(reset_btn)
        hint = QLabel("ภาพตัวอย่างทางขวาใช้ค่าในหน้านี้ทันที\nกด Save Settings ด้านล่างเพื่อใช้ค่าใหม่กับการตรวจจริง")
        hint.setStyleSheet("color:#777; font-size:11px;")
        form.addRow(hint)

        for w in (self.dist_nominal_spin, self.dist_tol_spin, self.min_bump_spin, self.min_width_spin):
            w.valueChanged.connect(self._on_judgement_changed)
        self.na_combo.currentIndexChanged.connect(self._on_judgement_changed)
        self.show_measure_check.toggled.connect(self._on_judgement_changed)
        self._update_range_label()
        return g

    def _build_preview_group(self) -> QGroupBox:
        g = QGroupBox("ภาพตัวอย่าง  (ชิ้นงานล่าสุดจากหน้า HOME)")
        v = QVBoxLayout(g)
        self.preview_label = ScaledImageLabel("ยังไม่มีภาพ - กด RUN ที่หน้า HOME ก่อน")
        self.preview_label.setStyleSheet("background-color:#111; color:#ccc; border:1px solid #444;")
        v.addWidget(self.preview_label, 1)
        self.preview_verdict = QLabel("")
        self.preview_verdict.setWordWrap(True)
        self.preview_verdict.setStyleSheet("font-weight:bold; font-size:14px;")
        v.addWidget(self.preview_verdict)
        nav = QHBoxLayout()
        self.preview_prev = QPushButton("◀  เก่ากว่า")
        self.preview_next = QPushButton("ใหม่กว่า  ▶")
        self.preview_info = QLabel("")
        self.preview_info.setAlignment(Qt.AlignCenter)
        self.preview_prev.clicked.connect(lambda: self._move_preview(+1))
        self.preview_next.clicked.connect(lambda: self._move_preview(-1))
        nav.addWidget(self.preview_prev)
        nav.addWidget(self.preview_info, 1)
        nav.addWidget(self.preview_next)
        v.addLayout(nav)
        return g

    def _collect_oring_settings(self) -> OringSettings:
        """The judgement settings as shown on this page (engine / send_per are not in the UI: kept from the file)."""
        base = self.oring_settings
        return OringSettings(
            engine=base.engine,
            dist_nominal_px=self.dist_nominal_spin.value(),
            dist_tol_px=self.dist_tol_spin.value(),
            min_bump_px=self.min_bump_spin.value(),
            min_width_px=self.min_width_spin.value(),
            na_send_as=self.na_combo.currentData() or base.na_send_as,
            send_per=base.send_per,
            show_measure=self.show_measure_check.isChecked(),
        ).clean()

    def _reset_judgement_defaults(self):
        d = OringSettings()
        self.dist_nominal_spin.setValue(d.dist_nominal_px)
        self.dist_tol_spin.setValue(d.dist_tol_px)
        self.min_bump_spin.setValue(d.min_bump_px)
        self.min_width_spin.setValue(d.min_width_px)
        self.na_combo.setCurrentIndex(max(0, self.na_combo.findData(d.na_send_as)))
        self.show_measure_check.setChecked(d.show_measure)

    def _update_range_label(self):
        lo, hi = self._collect_oring_settings().dist_range
        self.dist_range_label.setText(f"→ OK เมื่อแหวนอยู่ห่างปลายท่อ {lo} – {hi} px")

    def _on_judgement_changed(self, *_):
        self._update_range_label()
        self._refresh_preview()

    # ---- latest parts from HomePage ----
    def on_parts_evaluated(self, items: list):
        """Slot of HomePage.parts_evaluated: keep the latest crops for the preview and the measured numbers."""
        for it in reversed(items):                   # so that part #1 of the frame ends up first (newest)
            self._recent.appendleft(dict(crop=it["crop"], label=f"ชิ้น #{it['part']}"))
            for side in "LR":
                d = it.get("info", {}).get(side, {})
                if "dist" in d:
                    self._leg_stats.append(dict(dist=d["dist"], bump=d["bump"], width=d["width"]))
        if self._preview_pos > 0:                    # keep showing the same part while the user browses
            self._preview_pos = min(self._preview_pos + len(items), max(0, len(self._recent) - 1))
        if items:
            last = items[0].get("info", {})
            parts = []
            for side in "LR":
                d = last.get(side, {})
                parts.append(f"{side}: ระยะ {d['dist']:.0f} / นูน {d['bump']:.0f}" if "dist" in d else f"{side}: -")
            txt = "ค่าที่วัดล่าสุด:  " + "   ".join(parts)
            if self._leg_stats:
                dist = np.array([x["dist"] for x in self._leg_stats]); bump = np.array([x["bump"] for x in self._leg_stats])
                txt += (f"\n{len(self._leg_stats)} ขาล่าสุด: ระยะ {dist.min():.0f}–{dist.max():.0f} (เฉลี่ย {dist.mean():.0f}), "
                        f"นูน {bump.min():.0f}–{bump.max():.0f}")
            self.recent_label.setText(txt)
        self._refresh_preview()

    def _move_preview(self, delta: int):
        if self._recent:
            self._preview_pos = min(max(self._preview_pos + delta, 0), len(self._recent) - 1)
            self._refresh_preview()

    def _refresh_preview(self):
        """Re-judge the selected recent part with the values currently shown on this page and draw it."""
        n = len(self._recent)
        self.preview_prev.setEnabled(n > 0 and self._preview_pos < n - 1)
        self.preview_next.setEnabled(n > 0 and self._preview_pos > 0)
        if not n:
            return
        self._preview_pos = min(self._preview_pos, n - 1)
        item = self._recent[self._preview_pos]
        vis, cls, info = find_oring_v2(item["crop"], self._collect_oring_settings().to_params())
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
        self.preview_label.set_source(QPixmap.fromImage(qimg))
        st = info.get("status", {})
        reasons = [f"{s}: {info[s]['reason']}" for s in "LR" if info.get(s, {}).get("reason")]
        color = {"OK": "#1b8a1b", "NG": "#c62828", "NA": "#b8860b"}.get(cls, "#333")
        self.preview_verdict.setStyleSheet(f"font-weight:bold; font-size:14px; color:{color};")
        self.preview_verdict.setText(f"ผล: {cls}   (ซ้าย {st.get('L', '-')}, ขวา {st.get('R', '-')})"
                                     + (("\n" + " | ".join(reasons)) if reasons else ""))
        self.preview_info.setText(f"{item['label']}   ({self._preview_pos + 1}/{n})")

    # --- Methods from IoPage ---
    def _create_row(self, row_index: int, key: str):
        """Create one row for given key (row_index: 1-based, row 0 is header)."""
        name_lbl = QLabel(f"{key.upper()} :")
        name_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        name_lbl.setFixedWidth(80)

        status_lbl = QLabel(self.data_default)
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setFixedWidth(48)
        status_lbl.setStyleSheet("background-color:#CCCCCC; color:#000; border:1px solid #666; padding:2px;")

        arrow_lbl = QLabel("→")
        arrow_lbl.setFixedWidth(18)
        arrow_lbl.setAlignment(Qt.AlignCenter)

        plc_addr = self._plc_addr_for_index(row_index)
        if self.data_default == "OK":
            init_plc_val = 1
        elif self.data_default == "NG":
            init_plc_val = 0
        else:
            init_plc_val = -1
        plc_lbl = QLabel(f"{plc_addr}: {init_plc_val}")
        plc_lbl.setFixedWidth(60)

        self.group_layout.addWidget(name_lbl, row_index, 0, Qt.AlignLeft)
        self.group_layout.addWidget(status_lbl, row_index, 1, Qt.AlignLeft)
        self.group_layout.addWidget(arrow_lbl, row_index, 2, Qt.AlignCenter)
        self.group_layout.addWidget(plc_lbl, row_index, 3, Qt.AlignLeft)

        self.widgets[key] = (name_lbl, status_lbl, arrow_lbl, plc_lbl)
        self.states.setdefault(key, self.data_default)

    def _remove_row(self, key: str):
        """Remove a row's widgets and state."""
        widgets = self.widgets.pop(key, None)
        if not widgets:
            return
        for w in widgets:
            try:
                self.group_layout.removeWidget(w)
                w.deleteLater()
            except Exception:
                pass
        self.states.pop(key, None)

    def _ensure_row_for_key(self, key: str):
        """Ensure a row exists for key like 'data3'. Create if beyond current count."""
        try:
            idx = int(key.replace("data", "").replace("oring", ""))
        except Exception:
            return
        if key in self.widgets:
            return
        for i in range(len(self.widgets) + 1, idx + 1):
            new_key = f"data{i}"
            self._create_row(i, new_key)
        if idx > self.data_count:
            self.data_count = idx
            self.spin_count.blockSignals(True)
            self.spin_count.setValue(self.data_count)
            self.spin_count.blockSignals(False)
        self._adjust_group_height()

    def _plc_addr_for_index(self, index: int) -> str:
        """Return PLC address string for a data index (1-based)."""
        addr_num = self.plc_addr_start + (index - 1) * self.plc_addr_step
        return f"D{addr_num}"

    def set_data_count(self, new_count: int):
        """Slot: adjust UI rows to match new_count."""
        new_count = max(1, int(new_count))
        current = len(self.widgets)
        if new_count == current:
            return
        if new_count > current:
            for i in range(current + 1, new_count + 1):
                self._create_row(i, f"data{i}")
        else:
            for i in range(current, new_count, -1):
                self._remove_row(f"data{i}")
        self.data_count = new_count
        self._adjust_group_height()

    def _adjust_group_height(self):
        self.group_layout.invalidate()
        self.group.adjustSize()
        self.group.updateGeometry()
        hint_h = self.group.sizeHint().height()
        max_h = 600
        self.group.setMaximumHeight(min(max(hint_h, 60), max_h))

    def set_row_spacing(self, spacing: int):
        """Update vertical spacing (px) between data rows and refresh layout."""
        self.row_spacing = max(4, int(spacing))
        self.group_layout.setVerticalSpacing(max(4, self.row_spacing // 2))
        self._adjust_group_height()

    def set_data_default(self, default: str):
        """Change default value used for newly created rows (does not modify existing rows)."""
        self.data_default = (default or "NA").upper()

    def apply_default_to_existing(self):
        """Apply current data_default to all existing rows (updates UI + PLC labels)."""
        if not self.data_default:
            return
        updates = {}
        for key in list(self.states.keys()):
            updates[key] = self.data_default
        if updates:
            self._apply_statuss_loop(updates)

    def update_oring(self, s1: str = None, s2: str = None):
        """Backward-compatible: update two DATA entries (maps oring1->data1)."""
        updates = {}
        if s1 is not None:
            updates["data1"] = (s1 or "NA").upper()
        if s2 is not None:
            updates["data2"] = (s2 or "NA").upper()
        if updates:
            self._apply_statuss_loop(updates)

    def parse_and_update(self, summary: str):
        """
        Accepts strings like:
            "data1 OK | data2 NG"
            "oring1 ok | data3 NG"   (supports 'oringN' legacy)
        or JSON: '{"data1":"OK","data2":"NG"}'
        Will create rows if needed and update UI.
        """
        if not summary:
            return

        import json
        s = summary.strip()
        parsed = {}
        if s.startswith("{") and s.endswith("}"):
            try:
                j = json.loads(s)
                for k, v in j.items():
                    kk = str(k).lower()
                    if kk.startswith("oring"):
                        kk = kk.replace("oring", "data")
                    parsed[kk] = (str(v).upper() if v is not None else "NA")
            except Exception:
                return
        else:
            parts = [p.strip() for p in s.split("|")]
            for part in parts:
                if not part:
                    continue
                toks = part.split()
                if len(toks) < 2:
                    continue
                k = toks[0].lower()
                v = toks[1]
                if k.startswith("oring"):
                    k = k.replace("oring", "data")
                parsed[k] = (v.upper() if v is not None else "NA")

        if not parsed:
            return

        for key in list(parsed.keys()):
            self._ensure_row_for_key(key)

        self._apply_statuss_loop(parsed)

    def _apply_statuss_loop(self, updates: dict):
        """Internal loop to apply status updates to UI widgets."""
        for key, new_status in updates.items():
            if key not in self.widgets or self.states.get(key) == new_status:
                continue

            _name_lbl, status_lbl, _arrow_lbl, plc_lbl = self.widgets[key]
            self.states[key] = new_status

            bg_color = "#CCCCCC"  # default grey for NA
            plc_val = -1
            if new_status == "OK":
                bg_color = "#90EE90"  # light green
                plc_val = 1
            elif new_status == "NG":
                bg_color = "#F08080"  # light coral
                plc_val = 0

            status_lbl.setText(new_status)
            status_lbl.setStyleSheet(f"background-color:{bg_color}; color:#000; border:1px solid #666; padding:2px;")

            try:
                idx = int(key.replace("data", ""))
                plc_addr = self._plc_addr_for_index(idx)
                plc_lbl.setText(f"{plc_addr}: {plc_val}")
            except ValueError:
                # This will catch errors if the key is not in the format "data<number>"
                # and prevent the program from crashing.
                pass

    # --- new helpers for serial ---
    def populate_serial_ports(self):
        ports = []
        if list_ports:
            try:
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []
        if not ports:
            # fallback glob common linux device names
            ports = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
        current = self.serial_port_combo.currentText()
        self.serial_port_combo.blockSignals(True)
        self.serial_port_combo.clear()
        if ports:
            self.serial_port_combo.addItems(ports)
        else:
            self.serial_port_combo.addItem("")
        # restore selection if still present
        if current and current in ports:
            idx = self.serial_port_combo.findText(current)
            if idx >= 0:
                self.serial_port_combo.setCurrentIndex(idx)
        self.serial_port_combo.blockSignals(False)

    def _apply_serial_config(self):
        if not self.config.has_section('serial'):
            return
        sec = self.config['serial']
        port = sec.get('port', '')
        self.populate_serial_ports()
        if port:
            idx = self.serial_port_combo.findText(port)
            if idx >= 0:
                self.serial_port_combo.setCurrentIndex(idx)
            else:
                # add custom port if not listed
                self.serial_port_combo.addItem(port)
                self.serial_port_combo.setCurrentText(port)
        self.serial_baud_combo.setCurrentText(sec.get('baudrate','9600'))
        self.serial_parity_combo.setCurrentText(sec.get('parity','NONE').upper())
        self.serial_stopbits_combo.setCurrentText(sec.get('stopbits','1'))
        self.serial_bytesize_combo.setCurrentText(sec.get('bytesize','8'))
        try:
            self.modbus_slave_spin.setValue(int(sec.get('slave_id','1')))
        except ValueError:
            pass

    def test_serial_connection(self):
        port = self.serial_port_combo.currentText().strip()
        if not port:
            QMessageBox.information(self, "Serial Test", "No port selected.")
            return
        if serial is None:
            QMessageBox.warning(self, "Serial Test", "pyserial not installed (pip install pyserial).")
            return
        baud = int(self.serial_baud_combo.currentText())
        parity_map = {
            "NONE": serial.PARITY_NONE,
            "EVEN": serial.PARITY_EVEN,
            "ODD": serial.PARITY_ODD,
            "MARK": serial.PARITY_MARK,
            "SPACE": serial.PARITY_SPACE
        }
        stopbits_map = {"1": serial.STOPBITS_ONE, "1.5": serial.STOPBITS_ONE_POINT_FIVE, "2": serial.STOPBITS_TWO}
        bytesize_map = {"7": serial.SEVENBITS, "8": serial.EIGHTBITS}
        try:
            ser = serial.Serial(
                port=port,
                baudrate=baud,
                parity=parity_map.get(self.serial_parity_combo.currentText(), serial.PARITY_NONE),
                stopbits=stopbits_map.get(self.serial_stopbits_combo.currentText(), serial.STOPBITS_ONE),
                bytesize=bytesize_map.get(self.serial_bytesize_combo.currentText(), serial.EIGHTBITS),
                timeout=1
            )
            if ser.is_open:
                ser.close()
            QMessageBox.information(self, "Serial Test", f"Open OK: {port}")
        except Exception as e:
            QMessageBox.critical(self, "Serial Test", f"Failed: {e}")

    def update_from_classification(self, classification: str):
        """Map machine vision classification to DATA1 / DATA2.
        DATA1 shows OK status when classification == OK else NA.
        DATA2 shows NG status when classification == NG else NA.
        """
        classification = (classification or "NA").upper()
        updates = {}
        if "data1" not in self.widgets:
            self._ensure_row_for_key("data1")
        if "data2" not in self.widgets:
            self._ensure_row_for_key("data2")

        if classification == "OK":
            updates["data1"] = "OK"
            updates["data2"] = "NA"
        elif classification == "NG":
            updates["data1"] = "NA"
            updates["data2"] = "NG"
        else:
            updates["data1"] = "NA"
            updates["data2"] = "NA"
        self._apply_statuss_loop(updates)