#!/usr/bin/env python3
"""Interactive viewer: browse every saved frame of a report month and see
input (with matched box) | rectified crop + oring_v2 result | old saved output.

  cd machine_vision
  ubend/bin/python oring_test/viewer_oring_v2.py                       # default: <data_dir>/report/2026/09/report.csv
  ubend/bin/python oring_test/viewer_oring_v2.py --csv ~/UbendData/report/2026/09/report.csv --template model10.png

Keys: Left/Right (or A/D) = prev/next   PgUp/PgDn = -/+50   Home/End   Space = next old-NG
Matching runs in a background thread (neighbouring frames are prefetched); the v2 detector
re-runs instantly when you change the parameter boxes.
"""
import argparse, csv, os, queue, sys, threading
sys.dont_write_bytecode = True   # importing app/matching2.py must not rewrite the tracked .pyc files
from collections import OrderedDict

import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QSizePolicy)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from oring_v2 import Params, find_oring_v2  # noqa: E402
from app import storage  # noqa: E402
from test_oring_v2 import Cropper, frame_class, montage  # noqa: E402
from app import model_store  # noqa: E402

PREVIEW_W = 900


def to_pixmap(bgr):
    if bgr is None:
        return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class ImageView(QLabel):
    def __init__(self, title):
        super().__init__()
        self.title, self.pix = title, QPixmap()
        self.setMinimumSize(120, 120)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background:#202020;color:#aaa")
        self.setText(title)

    def set_image(self, bgr, text=None):
        self.pix = to_pixmap(bgr)
        if self.pix.isNull():
            self.setText(text or self.title)
        self._fit()

    def _fit(self):
        if not self.pix.isNull():
            self.setPixmap(self.pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        self._fit()
        super().resizeEvent(e)


class Worker(QObject):
    """Owns the native matcher; handles requests newest-first so the current frame wins over prefetch."""
    done = Signal(str, int, object)  # template, row index, result-dict-or-None

    def __init__(self):
        super().__init__()
        self.q = queue.LifoQueue()
        self.croppers = {}
        self.pending = set()
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def request(self, tpl, idx, path):
        with self.lock:
            if (tpl, idx) in self.pending:
                return
            self.pending.add((tpl, idx))
        self.q.put((tpl, idx, path))

    def run(self):
        while True:
            tpl, idx, path = self.q.get()
            res = None
            try:
                if tpl not in self.croppers:
                    self.croppers[tpl] = Cropper(model_store.resolve_model_path(tpl))
                frame = cv2.imread(path, cv2.IMREAD_COLOR)
                if frame is not None:
                    parts = self.croppers[tpl].crop_all(frame)
                    sc = PREVIEW_W / frame.shape[1]
                    prev = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                    for k, pt in enumerate(parts, 1):
                        q = (pt["quad"] * sc).astype(np.int32)
                        cv2.polylines(prev, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                        cv2.putText(prev, f"#{k}", (int(q[:, 0].min()), max(18, int(q[:, 1].min()) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    res = dict(parts=parts, preview=prev)
            except Exception as e:  # keep the viewer alive
                res = dict(parts=[], preview=None, error=str(e))
            with self.lock:
                self.pending.discard((tpl, idx))
            self.done.emit(tpl, idx, res)


class Viewer(QWidget):
    def __init__(self, rows, template):
        super().__init__()
        self.all_rows, self.rows, self.cur = rows, rows, 0
        self.cache = OrderedDict()
        self.worker = Worker()
        self.worker.done.connect(self.on_done)
        self.setWindowTitle("O-ring v2 viewer")
        self.resize(1700, 900)

        self.views = [ImageView("input + match"), ImageView("crop + v2 result"), ImageView("old output (saved)")]
        imgs = QHBoxLayout()
        for v, stretch in zip(self.views, (3, 3, 2)):
            col = QVBoxLayout()
            cap = QLabel(v.title); cap.setAlignment(Qt.AlignCenter)
            col.addWidget(cap); col.addWidget(v, 1)
            imgs.addLayout(col, stretch)

        self.filter = QComboBox(); self.filter.addItems(["All", "old OK", "old NG", "old NA"])
        self.tpl = QComboBox(); self.tpl.addItems(list(reversed(model_store.list_models())))
        if template in [self.tpl.itemText(i) for i in range(self.tpl.count())]:
            self.tpl.setCurrentText(template)
        self.prev_b, self.next_b = QPushButton("◀ Prev"), QPushButton("Next ▶")
        self.slider = QSlider(Qt.Horizontal)
        self.jump = QSpinBox()
        self.count_l = QLabel()
        self.bump = QDoubleSpinBox(); self.bump.setRange(0, 50); self.bump.setValue(Params.min_bump_px); self.bump.setSingleStep(0.5)
        self.dist = QDoubleSpinBox(); self.dist.setRange(0, 200); self.dist.setValue(Params.min_dist_px)
        self.width_ = QDoubleSpinBox(); self.width_.setRange(0, 400); self.width_.setValue(Params.min_width_px)
        self.th = QSpinBox(); self.th.setRange(5, 80); self.th.setValue(Params.tophat_rows)

        top = QHBoxLayout()
        for w in (QLabel("filter"), self.filter, QLabel("template"), self.tpl, self.prev_b, self.next_b,
                  self.slider, QLabel("go to #"), self.jump, self.count_l):
            top.addWidget(w)
        top.setStretchFactor(self.slider, 1)
        prm = QHBoxLayout()
        for lab, w in (("min bump px", self.bump), ("min dist px", self.dist), ("min width px", self.width_), ("tophat rows", self.th)):
            prm.addWidget(QLabel(lab)); prm.addWidget(w)
        prm.addStretch(1)

        self.info = QPlainTextEdit(); self.info.setReadOnly(True); self.info.setMaximumHeight(170)
        self.info.setFont(QFont("monospace", 10))

        lay = QVBoxLayout(self)
        lay.addLayout(top); lay.addLayout(imgs, 1); lay.addLayout(prm); lay.addWidget(self.info)

        self.filter.currentIndexChanged.connect(self.apply_filter)
        self.tpl.currentIndexChanged.connect(lambda _: self.show_cur())
        self.prev_b.clicked.connect(lambda: self.step(-1))
        self.next_b.clicked.connect(lambda: self.step(1))
        self.slider.valueChanged.connect(self.goto)
        self.jump.editingFinished.connect(lambda: self.goto(self.jump.value() - 1))
        for w in (self.bump, self.dist, self.width_, self.th):
            w.valueChanged.connect(lambda _: self.render())
        for w in (self.filter, self.tpl, self.slider):
            w.setFocusPolicy(Qt.NoFocus)  # keep arrow/page keys for the window itself
        self.apply_filter()

    # ---- navigation ----
    def apply_filter(self):
        f = self.filter.currentText()
        self.rows = self.all_rows if f == "All" else [r for r in self.all_rows if r["result"] == f.split()[1]]
        self.slider.blockSignals(True); self.slider.setRange(0, max(0, len(self.rows) - 1)); self.slider.blockSignals(False)
        self.jump.setRange(1, max(1, len(self.rows)))
        self.goto(0)

    def goto(self, i):
        if not self.rows:
            return
        self.cur = max(0, min(len(self.rows) - 1, int(i)))
        self.slider.blockSignals(True); self.slider.setValue(self.cur); self.slider.blockSignals(False)
        self.jump.blockSignals(True); self.jump.setValue(self.cur + 1); self.jump.blockSignals(False)
        self.show_cur()

    def step(self, d):
        self.goto(self.cur + d)

    def next_ng(self):
        for k in range(self.cur + 1, len(self.rows)):
            if self.rows[k]["result"] == "NG":
                return self.goto(k)

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_Right, Qt.Key_D): self.step(1)
        elif k in (Qt.Key_Left, Qt.Key_A): self.step(-1)
        elif k == Qt.Key_PageDown: self.step(50)
        elif k == Qt.Key_PageUp: self.step(-50)
        elif k == Qt.Key_Home: self.goto(0)
        elif k == Qt.Key_End: self.goto(len(self.rows) - 1)
        elif k == Qt.Key_Space: self.next_ng()
        else: super().keyPressEvent(e)

    # ---- data ----
    def key(self, i):
        return (self.tpl.currentText(), self.rows[i]["index"])

    def show_cur(self):
        if not self.rows:
            return
        for d in (2, 1, -1):  # prefetch first, current last (worker is LIFO)
            j = self.cur + d
            if 0 <= j < len(self.rows) and self.key(j) not in self.cache:
                self.worker.request(self.tpl.currentText(), int(self.rows[j]["index"]), self.rows[j]["input_image"])
        if self.key(self.cur) not in self.cache:
            r = self.rows[self.cur]
            self.worker.request(self.tpl.currentText(), int(r["index"]), r["input_image"])
        self.render()

    def on_done(self, tpl, idx, res):
        self.cache[(tpl, str(idx))] = res
        while len(self.cache) > 60:
            self.cache.popitem(last=False)
        if self.rows and self.key(self.cur) == (tpl, str(idx)):
            self.render()

    def render(self):
        if not self.rows:
            return
        r = self.rows[self.cur]
        old_out = cv2.imread(r["output_image"]) if os.path.isfile(r["output_image"]) else None
        self.views[2].set_image(old_out, "old output missing")
        res = self.cache.get(self.key(self.cur))
        head = f"#{r['index']}  {r['time']}  old={r['result']}   ({self.cur + 1}/{len(self.rows)})"
        self.count_l.setText(f"{self.cur + 1}/{len(self.rows)}")
        if res is None:
            self.views[0].set_image(None, "loading..."); self.views[1].set_image(None, "loading...")
            self.info.setPlainText(head + "\nmatching...")
            return
        self.views[0].set_image(res["preview"], "image unreadable")
        parts = res["parts"]
        if not parts:
            self.views[1].set_image(None, "no match" + (f": {res.get('error')}" if res.get("error") else ""))
            self.info.setPlainText(head + f"\nno part matched in this frame {res.get('error', '')}")
            return
        p = Params(min_bump_px=self.bump.value(), min_dist_px=self.dist.value(),
                   min_width_px=self.width_.value(), tophat_rows=self.th.value())
        visuals, classes, lines = [], [], []
        for k, pt in enumerate(parts, 1):
            vis, cls, info = find_oring_v2(pt["crop"], p)
            cv2.putText(vis, f"#{k}", (vis.shape[1] - 45, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            visuals.append(vis); classes.append(cls)
            st = info.get("status", {})
            lines.append(f"#{k} score={pt['score']:.2f} angle={pt['angle']:.0f}  {cls}  (L={st.get('L', '-')}, R={st.get('R', '-')})")
            for sd in "LR":
                d = info.get(sd, {})
                if "dist" in d:
                    lines.append(f"    {sd}: ring y={d['y']} x={d['x']} width={d['width']:.0f} (tube {d['tube_width']:.0f}) "
                                 f"dist-to-tip={d['dist']:.0f} bump={d['bump']:.1f} thick={d['thickness']}"
                                 + (f"  -> {d['reason']}" if d.get("reason") else ""))
                else:
                    lines.append(f"    {sd}: {d.get('reason', info.get('reason', '-'))}")
        self.views[1].set_image(visuals[0] if len(visuals) == 1 else montage(visuals, cols=2 if len(visuals) < 5 else 3))
        self.info.setPlainText(head + f"   parts={len(parts)}   NEW frame = {frame_class(classes)}\n" + "\n".join(lines))


def load_rows(path):
    rows = [r for r in storage.read_report_rows(path) if os.path.isfile(r["input_image"])]
    for r in rows:
        r["index"] = r["index"].strip()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(storage.report_dir() / "2026" / "09" / "report.csv"))
    ap.add_argument("--template", default="model10.png")
    a = ap.parse_args()
    rows = load_rows(a.csv)
    if not rows:
        sys.exit(f"no readable input images listed in {a.csv}")
    print(f"{len(rows)} frames from {a.csv}")
    app = QApplication(sys.argv)
    v = Viewer(rows, a.template)
    v.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
