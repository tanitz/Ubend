#!/usr/bin/env python3
"""Unit checks of app/ini_store.py, app/oring_settings.py and app/model_store.py (temporary folders only).

  cd machine_vision && ubend/bin/python tests/check_stores.py
"""
import os, shutil, sys, tempfile
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import ini_store, model_store
from app.oring_settings import OringSettings, load_settings, save_settings

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


tmp = tempfile.mkdtemp(prefix="ubend_stores_")
ini_store.BACKUP_DIR = os.path.join(tmp, "bak")
cfg = os.path.join(tmp, "config.ini")
open(cfg, "w").write("[find_oring]\ntemplate_file = model7.png\nh_oring = 40\n\n[serial]\nport = /dev/ttyUSB0\n")

# ---- ini_store ----
ini_store.update_ini("oring_v2", {"dist_nominal_px": 60}, cfg)
p = ini_store.read_ini(cfg)
check("update_ini keeps other sections / keys", p.get("find_oring", "template_file") == "model7.png" and p.get("serial", "port") == "/dev/ttyUSB0"
      and p.get("oring_v2", "dist_nominal_px") == "60")
check("write is atomic (no .tmp left)", not os.path.exists(cfg + ".tmp"))
for _ in range(14):
    ini_store.backup_ini(cfg, keep=10)
    import time; time.sleep(0.01)
n_bak = len([f for f in os.listdir(ini_store.BACKUP_DIR) if f.startswith("config.ini.")])
check("backup keeps only the newest 10 (and hides itself from git)", n_bak == 10 and os.path.exists(os.path.join(ini_store.BACKUP_DIR, ".gitignore")), str(n_bak))
check("reading a missing / corrupt file never raises", ini_store.read_ini(os.path.join(tmp, "nope.ini")).sections() == [])
bad = os.path.join(tmp, "bad.ini")
open(bad, "wb").write(b"\xff\xfe garbage [[[ \x00")
check("corrupt config -> default settings", load_settings(bad) == OringSettings())

# ---- OringSettings ----
s = OringSettings(dist_nominal_px=100, dist_tol_px=50)
p = s.to_params()
check("search window grows with the tolerance (nominal 100 +/- 50)", p.min_dist_px == 50 and p.max_dist_ok_px == 150 and p.max_dist_px >= 155, f"{p.min_dist_px} {p.max_dist_ok_px} {p.max_dist_px}")
check("defaults map to 40 - 70 px, bump 4", OringSettings().to_params().min_dist_px == 40 and OringSettings().to_params().max_dist_ok_px == 70 and OringSettings().to_params().min_bump_px == 4)
check("tolerance larger than nominal never gives a negative minimum", OringSettings(dist_nominal_px=10, dist_tol_px=50).to_params().min_dist_px == 0)
check("NA / NG / OK are sent as configured", OringSettings(na_send_as="NG").sent_value("NA") == "NG" and OringSettings(na_send_as="NA").sent_value("NA") == "NA"
      and OringSettings(na_send_as="NG").sent_value("ok") == "OK" and OringSettings(na_send_as="OK").sent_value("NA") == "OK")
saved = save_settings(OringSettings(dist_nominal_px=70, dist_tol_px=12, min_bump_px=5.5, min_width_px=30, na_send_as="NA", send_per="frame", show_measure=False), cfg)
check("save / load round trip", load_settings(cfg) == saved and saved.min_bump_px == 5.5 and saved.show_measure is False, str(load_settings(cfg)))
check("saving settings keeps the rest of the file", ini_store.read_ini(cfg).get("find_oring", "h_oring") == "40")
ini_store.update_ini("oring_v2", {"dist_nominal_px": "abc", "min_bump_px": "-4", "na_send_as": "zzz", "engine": "LEGACY"}, cfg)
bad_vals = load_settings(cfg)
check("hand-edited bad values are clamped / replaced", bad_vals.dist_nominal_px == 55 and bad_vals.min_bump_px == 0.0 and bad_vals.na_send_as == "NG" and bad_vals.engine == "legacy", str(bad_vals))

# ---- model_store ----
models = os.path.join(tmp, "model")
os.makedirs(models)
for n in ("model2.png", "model10.png", "model1.png", "readme.txt", ".hidden.png"):
    open(os.path.join(models, n), "wb").write(b"x")
real = model_store.MODEL_DIR
model_store.MODEL_DIR = models
try:
    check("list_models: natural order, only images, no hidden files", model_store.list_models() == ["model1.png", "model2.png", "model10.png"], str(model_store.list_models()))
    check("next_model_name continues the numbering", model_store.next_model_name() == "model11.png")
    src = os.path.join(tmp, "model2.png")
    open(src, "wb").write(b"y")
    check("import never overwrites an existing model", model_store.import_model(src) == "model2_2.png" and model_store.import_model(src) == "model2_3.png")
    check("resolve: bare name in model/, absolute path, unknown", model_store.resolve_model_path("model1.png") == os.path.join(models, "model1.png")
          and model_store.resolve_model_path(src) == src and model_store.resolve_model_path("nope.png") is None and model_store.resolve_model_path("") is None)
    check("resolve falls back to the old location (project root)", model_store.resolve_model_path("temp.jpg") == os.path.join(ROOT, "temp.jpg"))
    check("in_model_dir", model_store.in_model_dir(os.path.join(models, "x.png")) and not model_store.in_model_dir(os.path.join(tmp, "x.png")))
finally:
    model_store.MODEL_DIR = real
check("small-template rule (model7 247x216 small, model10 354x278 fine)", model_store.is_small((247, 216)) and not model_store.is_small((354, 278)))
shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
