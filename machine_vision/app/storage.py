"""Locations of generated data, independent of the source checkout.

UBEND_DATA_DIR overrides [storage] data_dir in app/config.ini. Image paths in
new report CSVs are relative to the CSV; legacy absolute paths remain readable.
"""
from __future__ import annotations

import configparser
import csv
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "app" / "config.ini"


def data_dir() -> Path:
    value = os.environ.get("UBEND_DATA_DIR", "").strip()
    if not value:
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(CONFIG_PATH, encoding="utf-8")
            value = parser.get("storage", "data_dir", fallback="").strip()
        except (configparser.Error, OSError, UnicodeError):
            value = ""
    path = Path(os.path.expandvars(value)).expanduser() if value else Path.home() / "UbendData"
    return (path if path.is_absolute() else ROOT_DIR / path).absolute()


def report_dir() -> Path:
    return data_dir() / "report"


def testing_dir(*parts: str) -> Path:
    return data_dir().joinpath("testing", *parts)


def resolve_report_image(reference: str, csv_path: str | Path) -> str:
    """Resolve CSV-relative images and old report/<year>/<month> paths.

    Keep unrelated references (including legacy base64 data) unchanged.
    A moved report's local image takes precedence over an old absolute path.
    """
    if not reference:
        return reference
    path = Path(reference)
    csv_dir = Path(csv_path).absolute().parent
    if not path.is_absolute():
        candidate = csv_dir / path
        return str(candidate) if candidate.is_file() else reference
    normalized = reference.replace("\\", "/")
    if "/report/" in normalized:
        suffix = normalized.rsplit("/report/", 1)[1]
        relocated = report_dir() / suffix
        # Also supports a report exported/moved independently of the data root.
        local = csv_dir / "image" / path.name
        for candidate in (local, relocated):
            if candidate.is_file():
                return str(candidate)
    return reference


def read_report_rows(csv_path: str | Path) -> list[dict]:
    """Read a path-based report for the offline tools, resolving its image paths."""
    with open(csv_path, newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for key in ("input_image", "output_image"):
            if key in row:
                row[key] = resolve_report_image(row[key], csv_path)
    return rows
