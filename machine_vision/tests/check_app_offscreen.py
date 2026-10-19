#!/usr/bin/env python3
"""End-to-end check of the main app WITHOUT a camera (offscreen Qt): real September frames are fed to HomePage.

  cd machine_vision && ubend/bin/python tests/check_app_offscreen.py [--shots DIR]

Uses a temporary copy of app/config.ini, a temporary report folder and a local UDP socket, so the real config / report are
untouched. Needs <data_dir>/report/2026/09 (frames) and model/model10.png. Prints PASS / FAIL per check; exit code 1 on any FAIL.
"""
import argparse, csv, os, shutil, socket, sys, tempfile, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication
from app import ini_store, model_store, storage
from main import MainWindow

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def recv_all(sock):
    out = []
    try:
        while True:
            out.append(sock.recvfrom(4096)[0].decode().strip())
    except socket.timeout:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", help="folder for screenshots of the HOME / SETTINGS pages")
    a = ap.parse_args()

    app = QApplication([])
    tmp = tempfile.mkdtemp(prefix="ubend_test_")
    cfg = os.path.join(tmp, "config.ini")
    shutil.copy(ini_store.CONFIG_PATH, cfg)
    report_dir = os.path.join(tmp, "report")
    os.makedirs(report_dir)
    ini_store.BACKUP_DIR = os.path.join(tmp, "config_backup")      # keep the repo clean

    real_report_csv = storage.report_dir() / "2026" / "09" / "report.csv"
    os.environ["UBEND_DATA_DIR"] = os.path.join(tmp, "data")
    win = MainWindow()
    win.resize(1700, 950)
    home, sett = win.home_page, win.setting_page
    sett.config_file = cfg
    home._get_report_month_dir = lambda: report_dir
    model_store.save_template_name = lambda name: ini_store.update_ini("find_oring", {"template_file": name}, cfg)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.3)
    home.tcp_enabled, home.tcp_host, home.tcp_port, home.tcp_mode = True, "127.0.0.1", rx.getsockname()[1], "plain"
    win.show()
    app.processEvents()

    rows = {r["index"]: r for r in storage.read_report_rows(real_report_csv)}

    def run_frame(idx):
        home.cv_frame = cv2.imread(rows[str(idx)]["input_image"])
        recv_all(rx)
        home.run_match_on_current_frame()
        app.processEvents()
        return recv_all(rx)

    # ---- model folder + drop-down ----
    check("template at start-up = model10.png", home.current_model_name == "model10.png", str(home.current_model_name))
    items = [home.model_combo.itemText(i) for i in range(home.model_combo.count())]
    check("MODEL list shows model/ files + Import entry", any(t.startswith("model10.png") for t in items) and "Import" in items[-1], str(items[:2]))
    check("MODEL list is in natural order (model2 before model10)",
          [t.split()[0] for t in items if t.startswith("model")][:3] == ["model0.png", "model1.png", "model2.png"])
    home.load_model("model7.png")
    check("switch to model7 persists to config + warns about the small template",
          ini_store.read_ini(cfg).get("find_oring", "template_file") == "model7.png" and "เล็ก" in home.result_label.text(), home.result_label.text()[:80])
    check("matcher re-created for the new template", home._part_matcher is not None and home._part_matcher.template_size == (247, 216))
    home.load_model("model10.png")
    check("back to model10", home.current_model_name == "model10.png" and home._part_matcher.template_size == (354, 278))
    check("unknown model does not break the selection", home.load_model("nope.png") is False and home.current_model_name == "model10.png")

    # ---- MODEL popup: about half as tall as the full list, scrollable with the mouse wheel ----
    combo = home.model_combo
    combo.showPopup()
    app.processEvents()
    view = combo.view()
    start = view.verticalScrollBar().value()
    app.sendEvent(view.viewport(), QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                                               Qt.NoModifier, Qt.NoScrollPhase, False))     # wheel up
    app.processEvents()
    popup_h = view.window().height()
    check("MODEL popup shows 6 rows (about half of the full list) with a visible scroll bar",
          combo.maxVisibleItems() == 6 and popup_h < 0.6 * combo.count() * view.sizeHintForRow(0) and view.verticalScrollBar().isVisible(),
          f"{popup_h}px for {combo.count()} entries")
    check("the mouse wheel scrolls the MODEL list", view.verticalScrollBar().value() != start, f"{start} -> {view.verticalScrollBar().value()}")
    combo.hidePopup()
    app.processEvents()

    # ---- matching + per-part counting / UDP / report ----
    msgs = run_frame(1783)                                   # 4 parts in one frame
    check("frame 1783: 4 parts all OK, counters +4, 4 UDP messages",
          home.counter_label.text().startswith("OK:4  NG:0  NA:0  Total:4") and [m.split(",")[1] for m in msgs] == ["OK"] * 4,
          home.counter_label.text() + " " + str([m.split(",")[1] for m in msgs]))
    with open(os.path.join(report_dir, "report.csv"), newline="") as f:
        rep = list(csv.DictReader(f))
    check("report: one row per part, the 4 parts share ONE input image",
          len(rep) == 4 and len({r["input_image"] for r in rep}) == 1 and len({r["output_image"] for r in rep}) == 4)
    msgs = run_frame(1938)                                   # a part without rings
    check("frame 1938 (no rings): NG counted and sent", [m.split(",")[1] for m in msgs] == ["NG"] and "NG:1" in home.counter_label.text(), home.result_label.text())
    msgs = run_frame(4118)
    check("frame 4118 (no part found): nothing counted, nothing sent", msgs == [] and home.counter_label.text().endswith("Total:5"), home.result_label.text())
    msgs = run_frame(1741)                                   # 3 parts, one of them NA
    sent = [m.split(",")[1] for m in msgs]
    shown = [r["cls"] for r in home.last_results]
    check("NA is sent as NG but still shown / counted as NA", "NA" in shown and sent.count("NG") >= 1 and "NA" not in sent and "NA:" in home.counter_label.text()
          and int(home.counter_label.text().split("NA:")[1].split()[0]) >= 1, f"shown={shown} sent={sent}")

    # ---- SETTINGS page: preview, tolerance, save, hot-apply ----
    check("SETTINGS page received the latest parts", len(sett._recent) >= 3 and "ค่าที่วัดล่าสุด" in sett.recent_label.text())
    run_frame(1783)
    sett._preview_pos = 0
    sett._refresh_preview()
    check("preview judges the latest part OK with the default settings", "ผล: OK" in sett.preview_verdict.text(), sett.preview_verdict.text())
    sett.dist_nominal_spin.setValue(20)
    sett.dist_tol_spin.setValue(5)
    check("preview follows the spin boxes at once (ring far outside 15-25 px -> NG)", "ผล: NG" in sett.preview_verdict.text(), sett.preview_verdict.text())
    check("range label shows the OK window", "15 – 25" in sett.dist_range_label.text(), sett.dist_range_label.text())
    check("home page still uses the saved settings before Save", home.oring_settings.dist_nominal_px == 55)
    sett.save_config()
    saved = ini_store.read_ini(cfg)
    check("Save writes [oring_v2] to config.ini", saved.get("oring_v2", "dist_nominal_px") == "20" and saved.get("oring_v2", "dist_tol_px") == "5")
    check("Save keeps the model chosen in HOME (file re-read before writing)", saved.get("find_oring", "template_file") == "model10.png")
    check("Save keeps the other sections (serial / report / tcp)", all(saved.has_section(x) for x in ("serial", "report", "tcp")))
    check("a backup of config.ini was made", os.path.isdir(ini_store.BACKUP_DIR) and any(f.startswith("config.ini.") for f in os.listdir(ini_store.BACKUP_DIR)))
    check("HOME applied the new settings at once", home.oring_settings.dist_nominal_px == 20 and home._params.max_dist_ok_px == 25.0)
    msgs = run_frame(1783)
    check("next frame is judged with the new tolerance (all 4 parts NG)", [m.split(",")[1] for m in msgs] == ["NG"] * 4, str([m.split(",")[1] for m in msgs]))
    sett._reset_judgement_defaults()
    check("reset button restores the defaults (55 +/- 15, bump 4)", sett.dist_nominal_spin.value() == 55 and sett.dist_tol_spin.value() == 15 and sett.min_bump_spin.value() == 4.0)

    # ---- legacy engine (fallback, config key engine = legacy) still works ----
    home.oring_settings.engine = "legacy"
    total_before = int(home.counter_label.text().split("Total:")[1])
    msgs = run_frame(1783)
    check("legacy engine: 4 parts judged, counted once each, no error, no OpenCV debug windows",
          not home.result_label.text().startswith("Error") and int(home.counter_label.text().split("Total:")[1]) == total_before + 4 and len(msgs) == 4,
          home.result_label.text()[:60])
    home.oring_settings.engine = "v2"

    # ---- MODEL: import a file / save a crop into model/ (temporary model folder, real one untouched) ----
    import app.ui.home_page as hp
    tmp_models = os.path.join(tmp, "model")
    os.makedirs(tmp_models)
    for f in ("model10.png", "model9.png"):
        shutil.copy(os.path.join("model", f), tmp_models)
    real_model_dir = model_store.MODEL_DIR
    model_store.MODEL_DIR = tmp_models
    try:
        ext = os.path.join(tmp, "my_template.png")
        shutil.copy(os.path.join("model", "model8.png"), ext)
        hp.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ext, ""))
        home._populate_models()
        home._on_model_activated(home.model_combo.count() - 1)            # the "Import from file..." entry
        check("Import copies the file into model/ and uses it", os.path.exists(os.path.join(tmp_models, "my_template.png"))
              and home.current_model_name == "my_template.png" and home.model_combo.currentData() == "my_template.png", str(home.current_model_name))
        check("next free model name is model11.png", model_store.next_model_name() == "model11.png", model_store.next_model_name())
        hp.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(tmp_models, model_store.next_model_name()), ""))
        home._switch_crop_button_to_save()
        home._save_matching_action()                                       # SAVE MATCHING after a crop
        check("SAVE MATCHING writes into model/ and selects the new model", os.path.exists(os.path.join(tmp_models, "model11.png"))
              and home.current_model_name == "model11.png" and home.model_combo.currentData() == "model11.png", str(home.current_model_name))
        check("the new model is remembered in config.ini", ini_store.read_ini(cfg).get("find_oring", "template_file") == "model11.png")
    finally:
        model_store.MODEL_DIR = real_model_dir
    home.load_model("model10.png")

    sett.save_config()                                        # defaults saved -> HOME is back to the normal judgement
    check("saving the defaults puts HOME back to 55 +/- 15", home.oring_settings.dist_nominal_px == 55 and home._params.max_dist_ok_px == 70.0)
    if a.shots:
        os.makedirs(a.shots, exist_ok=True)
        run_frame(1783)
        win.tabs.setCurrentIndex(0); app.processEvents(); home.grab().save(os.path.join(a.shots, "home.png"))
        win.is_unlocked, win.unlock_timestamp = True, time.time()      # SETTINGS tab is password protected (QInputDialog would block)
        win.tabs.setCurrentIndex(2); app.processEvents(); sett.grab().save(os.path.join(a.shots, "settings.png"))
        print("screenshots ->", a.shots)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}   (temp dir {tmp})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
