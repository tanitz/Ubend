#!/usr/bin/env python3
"""Move generated data outside the checkout, preserving files and backing up CSVs.

Run with the app stopped: python tools/migrate_data.py [--data-dir ~/UbendData]
The source and destination must be on the same filesystem (rename, no copying
or deleting images). Existing destination folders are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import storage


def snapshot(path):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {str(p.relative_to(path) if p != path else Path(".")):
            [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_dev, p.stat().st_ino] for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(storage.data_dir()))
    args = parser.parse_args()
    target = Path(args.data_dir).expanduser().resolve()
    if target == ROOT.parent or ROOT.parent in target.parents:
        parser.error("The data directory must be outside the source checkout")
    target.mkdir(parents=True, exist_ok=True)
    os.environ["UBEND_DATA_DIR"] = str(target)
    moves = [(ROOT / "report", target / "report"),
             (ROOT.parent / "audit-results", target / "audit"),
             (ROOT / "config_backup", target / "config_backup")]
    tools = ROOT / "oring_test"
    for path in sorted(tools.glob("*")):
        if path.name == "cache":
            moves.append((path, target / "testing" / "cache"))
        elif path.is_dir() and path.name.startswith(("out", "qa")):
            moves.append((path, target / "testing" / "results" / path.name))
        elif path.suffix == ".log":
            moves.append((path, target / "testing" / "logs" / path.name))
    pending = [(src, dst) for src, dst in moves if src.exists()]
    if not pending:
        print(f"No generated data left in the checkout; data root: {target}")
        return
    for src, dst in pending:
        if dst.exists():
            raise FileExistsError(f"Destination already exists; nothing moved: {dst}")
        if src.stat().st_dev != target.stat().st_dev:
            raise RuntimeError("Same filesystem required; nothing moved")
    run = target / "migration" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True)
    manifest = {"data_dir": str(target), "moves": [], "csv_changes": [], "references": {}}
    # Record missing references BEFORE moving, so existing missing images are
    # distinguishable from any migration regression.
    missing_before = set()
    for report in (ROOT / "report").glob("*/*/report.csv"):
        for row in storage.read_report_rows(report):
            for key in ("input_image", "output_image"):
                value = row.get(key, "")
                if value and not Path(value).is_file():
                    missing_before.add(value)
    (run / "missing-before.json").write_text(json.dumps(sorted(missing_before), ensure_ascii=False, indent=2))
    for src, dst in pending:
        before = snapshot(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        after = snapshot(dst)
        if after != before:
            raise RuntimeError(f"File metadata mismatch after rename: {dst}")
        manifest["moves"].append({"source": str(src), "destination": str(dst),
                                  "files": len(before), "bytes": sum(s[0] for s in before.values()),
                                  "verified_same_inodes": True})
        (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"MOVED {src.name}: {len(before)} files -> {dst}", flush=True)
    replacements = sorted([(str(src), str(dst)) for src, dst in moves], key=lambda p: -len(p[0]))

    def remap(value):
        for old, new in replacements:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value

    csvs = sorted((target / "report").glob("*/*/report.csv"))
    csvs += sorted((target / "testing").rglob("*.csv"))
    refs = missing = 0
    newly_missing = []
    for path in csvs:
        raw = path.read_bytes()
        with path.open(newline="", encoding="utf-8-sig") as source:
            rows = list(csv.reader(source))
        if not rows:
            continue
        header = rows[0]
        changed = 0
        is_report = path.parent.parent.parent == target / "report" and path.name == "report.csv"
        for row in rows[1:]:
            for i, value in enumerate(row):
                updated = remap(value)
                if is_report and i < len(header) and header[i] in ("input_image", "output_image") and value:
                    resolved = storage.resolve_report_image(updated, path)
                    if Path(resolved).is_file():
                        updated = os.path.relpath(resolved, path.parent)
                    refs += 1
                    if not Path(resolved).is_file():
                        missing += 1
                        # Original missing paths are compared before any path normalization.
                        if value not in missing_before:
                            newly_missing.append({"csv": str(path), "reference": value})
                if updated != value:
                    changed += 1
                    row[i] = updated
        if changed:
            backup = run / "csv-backups" / path.relative_to(target)
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(raw)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", newline="", encoding="utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8") as output:
                csv.writer(output, lineterminator="\r\n" if b"\r\n" in raw else "\n").writerows(rows)
                output.flush()
                os.fsync(output.fileno())
            shutil.copystat(path, tmp)
            tmp.replace(path)
            with path.open(newline="", encoding="utf-8-sig") as check:
                if list(csv.reader(check)) != rows:
                    raise RuntimeError(f"CSV verification failed: {path}")
            manifest["csv_changes"].append({"path": str(path), "rows": len(rows) - 1,
                                            "references_changed": changed, "backup": str(backup)})
    manifest["references"] = {"total": refs, "missing_before_unique": len(missing_before),
                              "missing_after": missing, "newly_missing": newly_missing}
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if newly_missing:
        raise RuntimeError(f"Newly missing images; inspect {run / 'manifest.json'}")
    print(f"VERIFIED {refs} report references; existing missing references: {missing}; new missing: 0")
    print(f"Manifest and original CSV backups: {run}")


if __name__ == "__main__":
    main()
