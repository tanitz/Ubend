#!/usr/bin/env python3
"""Offline tester for oring_v2.find_oring_v2 (does not touch the running app).

Examples (from machine_vision/):
  ubend/bin/python oring_test/test_oring_v2.py --csv ~/UbendData/report/2026/09/report.csv --limit 300 --only NG
  ubend/bin/python oring_test/test_oring_v2.py --input some_frame.jpg
  ubend/bin/python oring_test/test_oring_v2.py --input crops_dir --precropped

Outputs in --out (default <data_dir>/testing/results/out): results.csv, overlay/*.jpg (crop + v2 result),
sheet_*.jpg (contact sheets: old-vs-new disagreements).
Whole month, matcher once + QA sheets: build_cache.py (matcher -> crops cache), run_from_cache.py (detector -> qa/).
"""
import argparse, csv, glob, os, random, sys
sys.dont_write_bytecode = True   # importing app/matching2.py must not rewrite the tracked .pyc files
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from oring_v2 import Params, find_oring_v2  # noqa: E402
from app import storage  # noqa: E402
from app.part_matcher import PartMatcher  # noqa: E402

class Cropper(PartMatcher):
    """app.part_matcher.PartMatcher built from a template FILE (the offline tools work with file names).
    crop_all(frame) returns every part of the frame (NaN-safe, see part_matcher.py); .last holds the statistics."""
    def __init__(self, template_path, lib=None, score=0.55, max_parts=8):
        tpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            raise FileNotFoundError(template_path)
        super().__init__(tpl, lib=lib, score=score, max_parts=max_parts)

    def crop_with_quad(self, frame):
        """Best single part as (crop, quad, score, angle) or None (kept for the old single-part API)."""
        p = self.best(frame)
        return None if p is None else (p["crop"], p["quad"], p["score"], p["angle"])

    def __call__(self, frame):
        res = self.crop_with_quad(frame)
        return None if res is None else res[0]


def frame_class(classes):
    """Frame verdict from per-part verdicts: any NG -> NG, else any NA / no parts -> NA, else OK."""
    if "NG" in classes:
        return "NG"
    return "NA" if (not classes or "NA" in classes) else "OK"


def montage(images, cols=3, h=300):
    ims = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in images]
    w = max(i.shape[1] for i in ims)
    ims = [cv2.copyMakeBorder(i, 0, 0, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in ims]
    while len(ims) % cols:
        ims.append(np.full_like(ims[0], 255))
    return np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])


def collect(args):
    items = []  # (name, path, old_result)
    if args.csv:
        for r in storage.read_report_rows(args.csv):
            if os.path.isfile(r["input_image"]) and (not args.only or r["result"] == args.only):
                items.append((r["index"], r["input_image"], r["result"]))
        random.Random(args.seed).shuffle(items)
    if args.input:
        paths = [args.input] if os.path.isfile(args.input) else sorted(
            glob.glob(os.path.join(args.input, "*.jpg")) + glob.glob(os.path.join(args.input, "*.png")) +
            glob.glob(os.path.join(args.input, "*.bmp")))
        items += [(os.path.splitext(os.path.basename(p))[0], p, "") for p in paths]
    return items[:args.limit] if args.limit else items


def sheet(tiles, path, cols=4, w=420):
    if not tiles:
        return
    out = []
    for t, label in tiles:
        t = cv2.resize(t, (w, int(t.shape[0] * w / t.shape[1])))
        cv2.putText(t, label, (4, t.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        out.append(t)
    h = max(t.shape[0] for t in out)
    out = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in out]
    while len(out) % cols:
        out.append(np.full_like(out[0], 255))
    cv2.imwrite(path, np.vstack([np.hstack(out[i:i + cols]) for i in range(0, len(out), cols)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="report.csv (uses input_image + old result for comparison)")
    ap.add_argument("--input", help="image file or folder")
    ap.add_argument("--precropped", action="store_true", help="inputs are already rectified crops")
    ap.add_argument("--template", default=os.path.join(ROOT, "model", "model10.png"))
    ap.add_argument("--lib")
    ap.add_argument("--only", choices=["OK", "NG", "NA"], help="filter by old result (csv mode)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(storage.testing_dir("results", "out")))
    ap.add_argument("--min-bump", type=float); ap.add_argument("--min-dist", type=float)
    ap.add_argument("--min-width", type=float); ap.add_argument("--tophat-rows", type=int)
    a = ap.parse_args()
    if not (a.csv or a.input):
        ap.error("give --csv or --input")

    p = Params()
    for k, v in (("min_bump_px", a.min_bump), ("min_dist_px", a.min_dist),
                 ("min_width_px", a.min_width), ("tophat_rows", a.tophat_rows)):
        if v is not None:
            setattr(p, k, v)
    cropper = None if a.precropped else Cropper(a.template, a.lib)
    os.makedirs(os.path.join(a.out, "overlay"), exist_ok=True)

    items = collect(a)
    rows, tiles = [], {"old_ng_new_ok": [], "old_ok_new_ng": [], "new_ng": [], "na": []}
    print(f"found {len(items)} images, processing...", flush=True)
    for i, (name, path, old) in enumerate(items, 1):
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            rows.append(dict(name=name, part="", old=old, new="NA", reason="unreadable"))
            continue
        parts = [dict(crop=frame)] if a.precropped else cropper.crop_all(frame)
        if not parts:
            rows.append(dict(name=name, part="", old=old, new="NA", reason="no part matched"))
            continue
        visuals, classes, start = [], [], len(rows)
        for k, part in enumerate(parts, 1):
            vis, cls, info = find_oring_v2(part["crop"], p)
            cv2.putText(vis, f"#{k}", (vis.shape[1] - 45, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            visuals.append(vis); classes.append(cls)
            row = dict(name=name, part=k, old=old, new=cls, reason="", score=round(part.get("score", 0), 3))
            for side in "LR":
                d = info.get(side, {})
                for key in ("y", "x", "width", "dist", "bump", "thickness"):
                    row[f"{side}_{key}"] = round(d[key], 1) if key in d else ""
                if d.get("reason"):
                    row["reason"] += f"{side}:{d['reason']}; "
            rows.append(row)
        fcls = frame_class(classes)
        vis = visuals[0] if len(visuals) == 1 else montage(visuals)
        cv2.imwrite(os.path.join(a.out, "overlay", f"{name}.jpg"), vis)
        lab = f"{name} parts={len(parts)} old={old or '-'} new={fcls}"
        if old == "NG" and fcls == "OK": tiles["old_ng_new_ok"].append((vis, lab))
        elif old == "OK" and fcls == "NG": tiles["old_ok_new_ng"].append((vis, lab))
        elif fcls == "NG": tiles["new_ng"].append((vis, lab))
        elif fcls == "NA": tiles["na"].append((vis, lab))
        for r_ in rows[start:]:
            r_["frame_new"] = fcls

    keys = ["name", "part", "old", "new", "frame_new", "score", "reason"] + [f"{s}_{k}" for s in "LR" for k in ("y", "x", "width", "dist", "bump", "thickness")]
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval=""); w.writeheader(); w.writerows(rows)
    for k, t in tiles.items():
        sheet(t[:16], os.path.join(a.out, f"sheet_{k}.jpg"))

    frames = {}
    for r in rows:
        frames[r["name"]] = (r["old"], r.get("frame_new", r["new"]))
    parts_n = sum(1 for r in rows if r["part"] != "")
    print(f"processed {len(frames)} frames, {parts_n} parts -> {a.out}")
    print("  frame verdict (any NG part -> NG):")
    for o in ("OK", "NG", "NA", ""):
        sel = [v for v in frames.values() if v[0] == o]
        if sel:
            c = {k: sum(v[1] == k for v in sel) for k in ("OK", "NG", "NA")}
            print(f"  old={o or '-':2s} ({len(sel):4d}) -> new OK {c['OK']}  NG {c['NG']}  NA {c['NA']}")
    multi = sum(1 for n_ in frames if sum(1 for r in rows if r["name"] == n_ and r["part"] != "") > 1) if len(rows) < 3000 else None
    if multi is not None:
        print(f"  frames with more than one part: {multi}")
    for s in "LR":
        v = np.array([r[f"{s}_dist"] for r in rows if r.get(f"{s}_dist", "") != ""], float)
        b = np.array([r[f"{s}_bump"] for r in rows if r.get(f"{s}_bump", "") != ""], float)
        if v.size:
            q = lambda x: " ".join(f"{np.percentile(x, i):.0f}" for i in (1, 10, 50, 90, 99))
            print(f"  {s}: dist p1/10/50/90/99 = {q(v)} | bump = {q(b)}")


if __name__ == "__main__":
    main()
