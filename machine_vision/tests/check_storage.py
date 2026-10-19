"""Storage relocation checks; temporary files only, no camera or production writes."""
import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import storage


class StorageChecks(unittest.TestCase):
    def test_config_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "config.ini"
            cfg.write_text(f"[storage]\ndata_dir = {root / 'configured'}\n")
            with patch.object(storage, "CONFIG_PATH", cfg), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(storage.data_dir(), root / "configured")
                with patch.dict(os.environ, {"UBEND_DATA_DIR": str(root / "override")}):
                    self.assertEqual(storage.report_dir(), root / "override" / "report")

    def test_relative_csv_survives_moving_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            month = root / "original" / "report" / "2026" / "09"
            (month / "image").mkdir(parents=True)
            (month / "image" / "part.jpg").write_bytes(b"image fixture")
            with (month / "report.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["index", "input_image", "output_image"])
                writer.writerow(["1", "image/part.jpg", "image/part.jpg"])
            (root / "original").rename(root / "relocated")
            report = root / "relocated/report/2026/09/report.csv"
            row = storage.read_report_rows(report)[0]
            self.assertEqual(Path(row["input_image"]).read_bytes(), b"image fixture")

    def test_legacy_absolute_path_after_move(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"UBEND_DATA_DIR": tmp}):
            month = storage.report_dir() / "2025" / "09"
            (month / "image").mkdir(parents=True)
            image = month / "image" / "part.jpg"
            image.write_bytes(b"legacy fixture")
            old = "/old/checkout/machine_vision/report/2025/09/image/part.jpg"
            self.assertEqual(storage.resolve_report_image(old, month / "report.csv"), str(image))

    def test_moved_report_page_loads_thumbnail(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from app.ui.report_page import ReportPage
        import cv2
        import numpy as np

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"UBEND_DATA_DIR": tmp}):
            month = storage.report_dir() / "2026" / "09"
            (month / "image").mkdir(parents=True)
            cv2.imwrite(str(month / "image/part.png"), np.full((20, 20, 3), 120, np.uint8))
            report = month / "report.csv"
            report.write_text("index,time,result,input_image,output_image\n1,2026-09-01 12:00:00,OK,image/part.png,image/part.png\n")
            page = ReportPage()
            self.assertEqual(page._get_base_report_dir(), str(storage.report_dir()))
            _, rows = page._read_all_rows(str(report))
            self.assertFalse(page._pixmap_from_ref(rows[0][3]).isNull())
            page.close()
        app.processEvents()

    def test_home_writes_portable_paths_and_reloads_counters(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from app.ui.home_page import HomePage
        import configparser
        import numpy as np

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"UBEND_DATA_DIR": tmp}):
            home = HomePage()
            image = np.full((20, 30, 3), 100, np.uint8)
            with patch.object(configparser.ConfigParser, "getboolean", return_value=True):
                home._save_report_entry(image, image, "OK")
                home._save_report_entry(image, image, "NG")
            report = Path(home._get_report_month_dir()) / "report.csv"
            with report.open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertFalse(Path(rows[0]["input_image"]).is_absolute())
            self.assertTrue((report.parent / rows[0]["input_image"]).is_file())
            home._load_today_counters()
            self.assertEqual((home.count_ok, home.count_ng, home.count_total), (1, 1, 2))
            home.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
