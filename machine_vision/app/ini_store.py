"""Small helpers to read / update / write app/config.ini without clobbering what other pages wrote.

SettingPage, HomePage (selected model) and oring_settings all edit the same file; each of them must re-read it
right before it writes (never write a copy that was loaded at start-up), and the file is replaced atomically.
"""
from __future__ import annotations

import configparser
import os
import time
from app import storage

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
CONFIG_PATH = os.path.join(APP_DIR, "config.ini")          # same file SettingPage has always used
BACKUP_DIR = str(storage.data_dir() / "config_backup")


def read_ini(path: str = CONFIG_PATH) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    try:
        if os.path.exists(path):
            parser.read(path, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeDecodeError):
        parser = configparser.ConfigParser()
    return parser


def backup_ini(path: str = CONFIG_PATH, keep: int = 10) -> str | None:
    """Copy the file to config_backup/<name>.<timestamp> and keep only the newest `keep` copies."""
    if not os.path.exists(path):
        return None
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        keep_out = os.path.join(BACKUP_DIR, ".gitignore")         # the backups never show up in `git status`
        if not os.path.exists(keep_out):
            with open(keep_out, "w") as f:
                f.write("*\n")
        name = os.path.basename(path)
        now = time.time()
        dst = os.path.join(BACKUP_DIR, f"{name}.{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{int(now * 1000) % 1000:03d}")
        with open(path, "rb") as src, open(dst, "wb") as out:
            out.write(src.read())
        old = sorted(f for f in os.listdir(BACKUP_DIR) if f.startswith(name + "."))
        for f in old[:-keep]:
            os.remove(os.path.join(BACKUP_DIR, f))
        return dst
    except OSError:
        return None


def write_ini(parser: configparser.ConfigParser, path: str = CONFIG_PATH, backup: bool = False) -> None:
    if backup:
        backup_ini(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        parser.write(f)
    os.replace(tmp, path)


def update_ini(section: str, values: dict, path: str = CONFIG_PATH, backup: bool = False) -> None:
    """Read-modify-write: set `values` in [section], keep everything else in the file as it is on disk right now."""
    parser = read_ini(path)
    if not parser.has_section(section):
        parser.add_section(section)
    for k, v in values.items():
        parser.set(section, k, str(v))
    write_ini(parser, path, backup)
