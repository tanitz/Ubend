"""Template models: the files in machine_vision/model/ (png / jpg / bmp) and the one selected in app/config.ini.

[find_oring] template_file may be a bare file name (searched in model/ first, then in the old location = project
root), a path relative to the project root, or an absolute path - so config.ini files written before the model/
folder existed keep working.
"""
from __future__ import annotations

import os
import re
import shutil

import cv2

from app import ini_store

ROOT_DIR = ini_store.ROOT_DIR
MODEL_DIR = os.path.join(ROOT_DIR, "model")
EXTS = (".png", ".jpg", ".jpeg", ".bmp")
MIN_TEMPLATE_SIZE = (300, 250)     # the v2 detector needs the whole U-bend plus a margin inside the template footprint
DEFAULT_TEMPLATE = "model10.png"


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def ensure_dir() -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    return MODEL_DIR


def list_models() -> list[str]:
    """File names in model/, numbered names in natural order (model2 before model10)."""
    try:
        names = [f for f in os.listdir(ensure_dir())
                 if f.lower().endswith(EXTS) and not f.startswith(".") and os.path.isfile(os.path.join(MODEL_DIR, f))]
    except OSError:
        return []
    return sorted(names, key=_natural_key)


def resolve_model_path(name: str | None) -> str | None:
    """Absolute path of a template given as in config.ini, or None when it cannot be found."""
    if not name:
        return None
    name = name.strip()
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    for base in (MODEL_DIR, ROOT_DIR):
        p = os.path.abspath(os.path.join(base, name))
        if os.path.isfile(p):
            return p
    return None


def in_model_dir(path: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), MODEL_DIR]) == MODEL_DIR
    except ValueError:
        return False


def image_size(path: str | None) -> tuple[int, int]:
    """(width, height) of an image file, (0, 0) when it cannot be read."""
    img = cv2.imread(path) if path else None
    return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


def is_small(size: tuple[int, int]) -> bool:
    return size[0] < MIN_TEMPLATE_SIZE[0] or size[1] < MIN_TEMPLATE_SIZE[1]


def next_model_name(prefix: str = "model", ext: str = ".png") -> str:
    """model<N+1>.png where N is the highest number among model<N>.* in model/."""
    n = 0
    for f in list_models():
        m = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.\w+", f, re.I)
        if m:
            n = max(n, int(m.group(1)))
    return f"{prefix}{n + 1}{ext}"


def import_model(src: str) -> str:
    """Copy an image file into model/ (never overwrites: a free name is chosen) and return the new file name."""
    ensure_dir()
    base, ext = os.path.splitext(os.path.basename(src))
    name, k = base + ext, 1
    while os.path.exists(os.path.join(MODEL_DIR, name)):
        k += 1
        name = f"{base}_{k}{ext}"
    shutil.copy2(src, os.path.join(MODEL_DIR, name))
    return name


def save_template_name(name: str) -> None:
    """Remember the selected model (bare file name when it lives in model/) in app/config.ini [find_oring]."""
    ini_store.update_ini("find_oring", {"template_file": name})
