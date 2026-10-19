import os
import csv
import base64
from app import storage
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHBoxLayout, QMessageBox, QHeaderView, QAbstractItemView,
    QDateEdit, QTimeEdit, QSpinBox, QComboBox   # <-- added QComboBox
)
from PySide6.QtCore import Qt, QDateTime, QDate, QTime, QPoint, QSize, QRect, QEvent
from PySide6.QtGui import QPixmap, QImage, QCursor, QGuiApplication


class ReportPage(QWidget):
    """Report page with filters, modern light theme, and thumbnails in the table."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(8)

        # Light theme (white/bright) scoped to this page
        self.setStyleSheet(
            """
            * { font-family: 'Segoe UI', 'Noto Sans', sans-serif; }
            QWidget { background: #ffffff; color: #1f2937; }
            QLabel#TitleLabel { color: #111827; font-weight: 600; font-size: 22px; }
            QLabel#PathLabel { color: #6b7280; }

            QPushButton {
              background: #ffffff;
              border: 1px solid #d1d5db;
              padding: 6px 12px;
              border-radius: 6px;
            }
            QPushButton:hover { background: #f3f4f6; }
            QPushButton:pressed { background: #e5e7eb; }

            QTableWidget {
          background: #ffffff;
          /* ใช้สีเดียวกันทั้งแถว (ไม่แยก alternate) */
          alternate-background-color: #ffffff;
          gridline-color: #e5e7eb;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
        }
            QHeaderView::section {
              background: #f3f4f6;
              color: #111827;
              padding: 6px;
              border: 0px;
              border-bottom: 1px solid #e5e7eb;
              font-weight: 600;
            }
            QTableCornerButton::section { background: #f3f4f6; border: 0px; border-bottom: 1px solid #e5e7eb; }
            QTableView::item:selected { background: #e8f0fe; color: #111827; }

            QDateTimeEdit, QSpinBox {
              background: #ffffff;
              border: 1px solid #d1d5db;
              border-radius: 6px;
              padding: 4px 8px;
            }
            QDateTimeEdit:focus, QSpinBox:focus { border: 1px solid #60a5fa; }
            """
        )

        # Title
        title = QLabel("REPORT")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-size: 50px; font-weight: bold; color: #003366;")

        # Path label
        self.path_label = QLabel("")
        self.path_label.setObjectName("PathLabel")
        self.path_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # Do not show source path in UI by default
        self.path_label.setVisible(False)

        # Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("From:"))
        # Separate date and time controls
        self.dt_date = QDateEdit(self)
        self.dt_date.setDisplayFormat("yyyy-MM-dd")
        self.dt_date.setCalendarPopup(True)
        self.dt_date.setDate(QDate.currentDate())
        self.dt_date.setMinimumWidth(120)
        controls.addWidget(self.dt_date)

        controls.addWidget(QLabel("Time:"))
        self.dt_time = QTimeEdit(self)
        self.dt_time.setDisplayFormat("HH:mm:ss")
        # เดิมเป็นเวลาปัจจุบัน เปลี่ยนเป็น 00:00:00 เพื่อให้เห็นทั้งวัน
        self.dt_time.setTime(QTime(0, 0, 0))
        self.dt_time.setMinimumWidth(100)
        controls.addWidget(self.dt_time)

        controls.addWidget(QLabel("Count:"))
        self.count_spin = QSpinBox(self)
        self.count_spin.setRange(1, 1000)
        self.count_spin.setValue(10)
        self.count_spin.setMinimumWidth(80)
        controls.addWidget(self.count_spin)

        # --- added result filter ---
        controls.addWidget(QLabel("Result:"))
        self.result_combo = QComboBox(self)
        self.result_combo.addItems(["All", "OK", "NG"])
        self.result_combo.setMinimumWidth(90)
        controls.addWidget(self.result_combo)
        self.result_combo.currentIndexChanged.connect(self.apply_filter_from_controls)
        # ---------------------------

        self.search_btn = QPushButton("Search")
        self.search_btn.setMinimumHeight(34)
        self.search_btn.clicked.connect(self.apply_filter_from_controls)
        controls.addWidget(self.search_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setMinimumHeight(34)
        self.refresh_btn.clicked.connect(self.load_latest_entries)
        controls.addWidget(self.refresh_btn)
        controls.addStretch(1)

        # Table with thumbnails (Index, Time, Result, Input, Output)
        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(["Index", "Time", "Result", "Input", "Output"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(3, 160)
        self.table.setColumnWidth(4, 160)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(120)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # ปิดการใช้สีสลับแถว เพื่อให้ทุกแถวใช้สีเดียวกัน
        self.table.setAlternatingRowColors(False)

        # Assemble
        self.layout.addWidget(title)
        self.layout.addLayout(controls)
        self.layout.addWidget(self.path_label)
        self.layout.addWidget(self.table)

    # State
        self.all_rows = []
        self.header = []
        self.current_rows = []

    # Initial load
        self.load_latest_entries()

    # --- Data helpers ---
    def _get_base_report_dir(self) -> str:
        return str(storage.report_dir())

    def _find_latest_report_csv(self) -> str | None:
        base = self._get_base_report_dir()
        if not os.path.isdir(base):
            return None

        years = sorted(
            [d for d in os.listdir(base) if d.isdigit() and os.path.isdir(os.path.join(base, d))],
            reverse=True,
        )
        for y in years:
            ydir = os.path.join(base, y)
            months = sorted(
                [d for d in os.listdir(ydir) if d.isdigit() and os.path.isdir(os.path.join(ydir, d))],
                reverse=True,
            )
            for m in months:
                candidate = os.path.join(ydir, m, "report.csv")
                if os.path.isfile(candidate):
                    return candidate

        now = datetime.now()
        fallback = os.path.join(base, now.strftime("%Y"), now.strftime("%m"), "report.csv")
        return fallback if os.path.isfile(fallback) else None

    def _read_all_rows(self, csv_path: str):
        header: list[str] = []
        rows: list[list[str]] = []
        if not csv_path or not os.path.isfile(csv_path):
            return header, rows
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return header, rows
            for row in reader:
                if row:
                    for key in ("input_image", "output_image"):
                        if key in header:
                            column = header.index(key)
                            if column < len(row):
                                row[column] = storage.resolve_report_image(row[column], csv_path)
                    rows.append(row)
        return header, rows

    def _parse_time(self, s: str) -> datetime | None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        return None

    # --- Loading & filtering ---
    def load_latest_entries(self):
        csv_path = self._find_latest_report_csv()
        if not csv_path:
            self.path_label.setText("No report found")
            self.table.setRowCount(0)
            return
        self.path_label.setText(f"Source: {csv_path}")
        try:
            header, rows = self._read_all_rows(csv_path)
            self.header = header
            self.all_rows = rows
        except Exception as e:
            QMessageBox.warning(self, "Report", f"Failed to read report: {e}")
            self.all_rows = []
        # ใช้ตัวกรองปัจจุบันทันที
        self.apply_filter_from_controls()

    def apply_default_last_n(self):
        n = self.count_spin.value()
        rows = self.all_rows[-n:] if self.all_rows else []
        self._update_table(list(reversed(rows)))

    def apply_filter_from_controls(self):
        dt = QDateTime(self.dt_date.date(), self.dt_time.time()).toPython()
        n = self.count_spin.value()
        result_filter = self.result_combo.currentText().lower()

        candidates: list[tuple[datetime, list[str]]] = []
        for row in self.all_rows:
            if len(row) < 3:
                continue
            t = self._parse_time(row[1])
            if t is None:
                continue
            if t < dt:
                continue
            result_val = row[2].strip().lower()
            if result_filter != "all" and result_val != result_filter:
                continue
            candidates.append((t, row))

        # เรียงจากล่าสุด -> เก่าสุด
        candidates.sort(key=lambda x: x[0], reverse=True)

        # จำกัด N ล่าสุดตาม filter ที่เลือก
        filtered_rows = [r for _, r in candidates[:n]]

        # กรณีไม่พบอะไรเลยและ dt อาจถูกตั้งอนาคต -> fallback เอา N ล่าสุดแบบไม่ใช้ dt
        if not filtered_rows and candidates == [] and dt > datetime.now():
            all_by_time: list[tuple[datetime, list[str]]] = []
            for row in self.all_rows:
                if len(row) < 3:
                    continue
                t = self._parse_time(row[1])
                if t is None:
                    continue
                result_val = row[2].strip().lower()
                if result_filter != "all" and result_val != result_filter:
                    continue
                all_by_time.append((t, row))
            all_by_time.sort(key=lambda x: x[0], reverse=True)
            filtered_rows = [r for _, r in all_by_time[:n]]

        self._update_table(filtered_rows)
        # สรุป (ถ้าต้องการให้โชว์ เปิดบรรทัดล่าง)
        # self.path_label.setText(f"{len(filtered_rows)} shown / {len(candidates)} matched (filter={self.result_combo.currentText()})")

    # --- Table rendering ---
    def _update_table(self, rows: list[list[str]]):
        self.current_rows = rows
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            idx = row[0] if len(row) > 0 else ""
            t = row[1] if len(row) > 1 else ""
            result = row[2] if len(row) > 2 else ""
            self.table.setItem(r, 0, QTableWidgetItem(str(idx)))
            self.table.setItem(r, 1, QTableWidgetItem(str(t)))
            self.table.setItem(r, 2, QTableWidgetItem(str(result)))

            # Row height for thumbnails
            self.table.setRowHeight(r, max(self.table.verticalHeader().defaultSectionSize(), 120))

            in_ref = row[3] if len(row) > 3 else ""
            out_ref = row[4] if len(row) > 4 else ""
            in_label = HoverPreviewLabel(preview_max_size=QSize(800, 600))
            in_label.setAlignment(Qt.AlignCenter)
            in_label.setStyleSheet("background:#f8fafc; border:1px solid #e5e7eb; border-radius:6px;")
            out_label = HoverPreviewLabel(preview_max_size=QSize(800, 600))
            out_label.setAlignment(Qt.AlignCenter)
            out_label.setStyleSheet("background:#f8fafc; border:1px solid #e5e7eb; border-radius:6px;")

            target_h = self.table.rowHeight(r) - 8
            target_w = max(self.table.columnWidth(3), 140) - 8
            in_pix = self._pixmap_from_ref(in_ref)
            if not in_pix.isNull():
                # Set both the thumbnail and the original for preview
                in_label.set_thumbnail(in_pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                in_label.set_original(in_pix)
            else:
                in_label.setText("N/A")
            out_pix = self._pixmap_from_ref(out_ref)
            if not out_pix.isNull():
                out_label.set_thumbnail(out_pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                out_label.set_original(out_pix)
            else:
                out_label.setText("N/A")

            self.table.setCellWidget(r, 3, in_label)
            self.table.setCellWidget(r, 4, out_label)

    def _pixmap_from_ref(self, ref: str) -> QPixmap:
        pix = QPixmap()
        if not ref:
            return pix
        if os.path.exists(ref):
            pix.load(ref)
            return pix
        try:
            data = base64.b64decode(ref)
            img = QImage.fromData(data)
            if not img.isNull():
                pix = QPixmap.fromImage(img)
        except Exception:
            pass
        return pix


class _ImagePreviewWidget(QWidget):
    """Small, floating widget that shows a large image preview centered over the app window."""
    def __init__(self, parent=None):
        # Use a frameless tool window that stays on top so it appears like a centered popup
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # Ensure it never steals focus and doesn't activate the window
        self.setFocusPolicy(Qt.NoFocus)
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        # Do not intercept mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            "background: rgba(17,24,39,0.95); border:1px solid rgba(15,23,42,0.9); color:#f9fafb; padding:8px; border-radius:10px;"
        )
        self._pix = QPixmap()

        # Ensure widget does not get focus and does not steal input
        self.setFocusPolicy(Qt.NoFocus)
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus)

    def set_pixmap(self, pix: QPixmap):
        self._pix = pix
        self._label.setPixmap(pix)
        self._label.adjustSize()
        self.adjustSize()

    def sizeHint(self):
        return self._label.sizeHint()


class HoverPreviewLabel(QLabel):
    """QLabel that shows a larger image preview on mouse hover.

    - Use set_original(pixmap) to provide the full-size image for preview.
    - Use set_thumbnail(pixmap) to set the thumbnail shown inside the table cell.
    """

    def __init__(self, preview_max_size: QSize = QSize(800, 600), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._orig_pix = QPixmap()
        self._preview_max = preview_max_size
        self._preview = None  # type: _ImagePreviewWidget | None

        # Ensure preview is cleaned when this label disappears
        self.destroyed.connect(self._cleanup_preview)

    # Public API for clarity
    def set_original(self, pix: QPixmap):
        self._orig_pix = pix

    def set_thumbnail(self, pix: QPixmap):
        super().setPixmap(pix)

    # Events
    def enterEvent(self, event):
        super().enterEvent(event)
        self._show_preview()

    def leaveEvent(self, event):
        self._hide_preview()
        super().leaveEvent(event)

    def hideEvent(self, event):
        # If the cell or table hides (e.g., filter/refresh), close preview
        self._hide_preview()
        super().hideEvent(event)

    def mouseMoveEvent(self, event):
        self._move_preview()
        super().mouseMoveEvent(event)

    def eventFilter(self, obj, event):
        # Hide preview if the top-level window deactivates/closes/hides
        if obj is self.window() and event.type() in (QEvent.WindowDeactivate, QEvent.Hide, QEvent.Close):
            self._hide_preview()
        return super().eventFilter(obj, event)

    # Internals
    def _ensure_preview(self):
        if self._preview is None:
            # Parent the preview to the top-level window so it can be centered over the app
            top = self.window() if self.window() is not None else None
            self._preview = _ImagePreviewWidget(top)
            # Track window state to auto-hide preview when app deactivates
            if top:
                top.installEventFilter(self)

    def _hide_preview(self):
        if self._preview:
            self._preview.hide()
            # Also close to ensure it disappears even if this label is destroyed soon after
            self._preview.close()

    def _cleanup_preview(self):
        if self._preview:
            try:
                self._preview.hide()
                self._preview.close()
                self._preview.deleteLater()
            finally:
                self._preview = None

    def _target_geometry(self, pix: QPixmap) -> QPixmap:
        if pix.isNull():
            return pix
        # Fit into available screen up to self._preview_max, preserving aspect ratio
        screen = self.window().screen() if hasattr(self.window(), "screen") else QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else QRect(0, 0, self._preview_max.width(), self._preview_max.height())
        max_w = min(self._preview_max.width(), max(200, int(avail.width() * 0.6)))
        max_h = min(self._preview_max.height(), max(150, int(avail.height() * 0.6)))
        return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _show_preview(self):
        if self._orig_pix.isNull():
            return
        self._ensure_preview()
        big = self._target_geometry(self._orig_pix)
        self._preview.set_pixmap(big)
        self._position_near_cursor()
        self._preview.show()

    def _move_preview(self):
        if self._preview and self._preview.isVisible():
            self._position_near_cursor()

    def _position_near_cursor(self):
        if not self._preview:
            return
        # Center the preview over the application's top-level window
        top = self.window()
        if top:
            center_global = top.mapToGlobal(top.rect().center())
        else:
            center_global = QGuiApplication.primaryScreen().availableGeometry().center()
        w = self._preview.width()
        h = self._preview.height()
        x = center_global.x() - (w // 2)
        y = center_global.y() - (h // 2)
        # keep inside screen bounds
        screen = top.screen() if (top and hasattr(top, "screen")) else QGuiApplication.primaryScreen()
        if screen:
            avail: QRect = screen.availableGeometry()
            x = min(max(avail.left(), x), avail.right() - w)
            y = min(max(avail.top(), y), avail.bottom() - h)
        self._preview.move(QPoint(int(x), int(y)))
