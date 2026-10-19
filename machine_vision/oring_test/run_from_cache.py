#!/usr/bin/env python3
"""Stage 2 of the QA pipeline: run find_oring_v2 on the cached crops (no matcher, fast) and build the QA material.

  cd machine_vision
  ubend/bin/python oring_test/run_from_cache.py                              # cache/ -> qa/
  ubend/bin/python oring_test/run_from_cache.py --min-bump 5 --min-dist 45   # same Params overrides as test_oring_v2.py
  ubend/bin/python oring_test/run_from_cache.py --param seed_s=50 --indices 1783,4125 --qa /tmp/qa_try

Writes to --qa (default <data_dir>/testing/results/qa):
  results.csv         one row per part: index, part, old, new, frame_new, score, L_/R_ y,x,width,dist,bump,thickness,
                      reason (+ sheet, tile, note, L_/R_tilt = leg axis tilt in deg, L_/R_tube = tube width in px);
                      a frame without a match has one row with an empty part, new=NA
  frames_summary.csv  index, n_parts, frame_new, old
  sheets/sheet_0001.jpg ...   12 tiles per sheet (4 x 3, tile 354 x (278+28) px): the v2 overlay at crop size + a caption
                      strip '#<index> p<k> new=<OK|NG|NA> old=<..>'; tile number 1..12 at the top-left. A frame without a
                      match gets a tile with the downscaled input and 'NO MATCH'. Order: frame index, then part.
  sheets_index.csv    sheet, tile_no, index, part, new, old     (empty part = no-match tile)

--changed-vs OLD_RESULTS.csv  additionally writes <qa>/changed/ : only the tiles whose result differs from OLD_RESULTS.csv
                      (any measurement or verdict; new tiles count too), packed 12 per sheet, with changed/results.csv and
                      changed/sheets_index.csv (+ prev_new, diff_fields). Used to re-review only what a code change touched.
"""
import argparse, csv, dataclasses, multiprocessing as mp, os, shutil, sys, time
sys.dont_write_bytecode = True   # importing app/matching2.py must not rewrite the tracked .pyc files
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oring_v2 import Params, find_oring_v2  # noqa: E402
from app import storage  # noqa: E402
from test_oring_v2 import frame_class  # noqa: E402

TILE_W, IMG_H, CAP_H, COLS, ROWS = 354, 278, 28, 4, 3
TILE_H = IMG_H + CAP_H
PER_SHEET = COLS * ROWS
CAP_BG = {"OK": (215, 240, 215), "NG": (205, 205, 250), "NA": (190, 235, 250)}   # BGR tints of the caption strip
FONT = cv2.FONT_HERSHEY_SIMPLEX
SIDE_KEYS = ("y", "x", "width", "dist", "bump", "thickness")
RES_COLS = (["index", "part", "old", "new", "frame_new", "score"] + [f"{s}_{k}" for s in "LR" for k in SIDE_KEYS]
            + ["reason", "sheet", "tile", "note", "L_tilt", "R_tilt", "L_tube", "R_tube"])


def caption(text, new, width=TILE_W):
    """28 px strip with the text as large as fits."""
    strip = np.empty((CAP_H, width, 3), np.uint8)
    strip[:] = CAP_BG.get(new, (240, 240, 240))
    scale = 0.7
    while scale > 0.35 and cv2.getTextSize(text, FONT, scale, 2)[0][0] > width - 8:
        scale -= 0.05
    cv2.putText(strip, text, (5, 20), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return strip


def tile_number(tile, n):
    cv2.rectangle(tile, (0, 0), (33 if n < 10 else 44, 26), (0, 0, 0), -1)
    cv2.putText(tile, str(n), (4, 20), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def outlined(img, text, org, scale, color, thick=2):
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def fit(img, w, h):
    """Resize keeping the aspect ratio to fit w x h, centred on white."""
    sc = min(w / img.shape[1], h / img.shape[0])
    im = cv2.resize(img, (max(1, round(img.shape[1] * sc)), max(1, round(img.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 255, np.uint8)
    y, x = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
    out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


def analyse(item, params, cache):
    """-> (tile image 354 x 306, result row) for one cached part or no-match frame."""
    idx, part, old = item["index"], item["part"], item["old_result"]
    row = dict(index=idx, part=part, old=old, score=item["score"], note=item["note"])
    if part == "":
        img = cv2.imread(os.path.join(cache, "preview", f"{idx}.jpg"), cv2.IMREAD_COLOR)
        img = fit(img, TILE_W, IMG_H) if img is not None else np.full((IMG_H, TILE_W, 3), 255, np.uint8)
        outlined(img, "NO MATCH", (40, 24), 0.8, (0, 0, 255))
        row.update(new="NA", reason=item["note"] or "no match")
        cap = f"#{idx} NO MATCH old={old or '-'}"
        return np.vstack([img, caption(cap, "NA")]), row
    crop = cv2.imread(os.path.join(cache, item["crop_path"]), cv2.IMREAD_COLOR)
    if crop is None:
        img = np.full((IMG_H, TILE_W, 3), 255, np.uint8)
        outlined(img, "CROP MISSING", (40, 24), 0.8, (0, 0, 255))
        row.update(new="NA", reason="crop file missing")
    else:
        vis, cls, info = find_oring_v2(crop, params)
        img = vis if vis.shape[:2] == (IMG_H, TILE_W) else fit(vis, TILE_W, IMG_H)
        row["new"] = cls
        reasons = []
        for side in "LR":
            d = info.get(side, {})
            for k in SIDE_KEYS:
                row[f"{side}_{k}"] = round(d[k], 1) if k in d else ""
            row[f"{side}_tilt"] = round(d["tilt_raw"], 1) if "tilt_raw" in d else ""
            row[f"{side}_tube"] = round(d["tube_width"], 1) if "tube_width" in d else ""
            if d.get("reason"):
                reasons.append(f"{side}:{d['reason']}")
        if info.get("reason"):
            reasons.append(info["reason"])
        row["reason"] = "; ".join(reasons)
    cap = f"#{idx} p{part} new={row['new']} old={old or '-'}"
    return np.vstack([img, caption(cap, row["new"])]), row


def make_sheet(task):
    """task = (sheet_no, items, params, cache, out_dir) -> (sheet_no, rows). One sheet = 12 tiles."""
    no, items, params, cache, out = task
    sheet = np.full((ROWS * TILE_H, COLS * TILE_W, 3), 255, np.uint8)
    rows = []
    for k, item in enumerate(items):
        tile, row = analyse(item, params, cache)
        tile_number(tile, k + 1)
        cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (150, 150, 150), 1)
        r, c = divmod(k, COLS)
        sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile
        row.update(sheet=no, tile=k + 1)
        rows.append(row)
    cv2.imwrite(os.path.join(out, "sheets", f"sheet_{no:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return no, rows


def write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


CMP_COLS = [c for c in RES_COLS if c not in ("sheet", "tile", "frame_new", "note")]


def _same(a, b):
    a, b = str(a), str(b)
    try:
        return abs(float(a) - float(b)) < 0.051
    except ValueError:
        return a == b


def make_changed(a, rows, meta, p):
    """Render only the tiles whose result differs from a previous results.csv into <qa>/changed/."""
    old = {(r["index"], r["part"]): r for r in csv.DictReader(open(a.changed_vs, newline=""))}
    items = {(m["index"], m["part"]): m for m in meta}
    sel, info = [], []
    for r in rows:
        key = (str(r["index"]), str(r["part"]))
        o = old.get(key)
        diff = [c for c in CMP_COLS if o is None or not _same(r.get(c, ""), o.get(c, ""))]
        if diff:
            sel.append(items[key])
            info.append(dict(prev_new=o["new"] if o else "(new tile)", diff_fields="+".join(diff)))
    out = os.path.join(a.qa, "changed")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "sheets"))
    tasks = [(n + 1, sel[i:i + PER_SHEET], p, a.cache, out) for n, i in enumerate(range(0, len(sel), PER_SHEET))]
    crow = []
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for no, rs in pool.imap(make_sheet, tasks):
            crow += rs
    for r, i in zip(crow, info):
        r.update(i)
    write(os.path.join(out, "results.csv"), RES_COLS + ["prev_new", "diff_fields"], crow)
    write(os.path.join(out, "sheets_index.csv"), ["sheet", "tile_no", "index", "part", "new", "old", "prev_new", "diff_fields"],
          [dict(sheet=f"sheet_{r['sheet']:04d}", tile_no=r["tile"], index=r["index"], part=r["part"], new=r["new"], old=r["old"],
                prev_new=r["prev_new"], diff_fields=r["diff_fields"]) for r in crow])
    moves = Counter((r["prev_new"], r["new"]) for r in crow)
    print(f"\nchanged vs {a.changed_vs}: {len(crow)} of {len(rows)} tiles -> {len(tasks)} sheets in {out}")
    for (pv, nw), n in sorted(moves.items(), key=lambda kv: -kv[1]):
        print(f"  {pv:>10s} -> {nw}: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(storage.testing_dir("cache")))
    ap.add_argument("--qa", default=str(storage.testing_dir("results", "qa")))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--indices", help="only these frame indices (comma separated)")
    ap.add_argument("--limit", type=int, help="only the first N frames")
    ap.add_argument("--min-bump", type=float); ap.add_argument("--min-dist", type=float)
    ap.add_argument("--min-width", type=float); ap.add_argument("--tophat-rows", type=int)
    ap.add_argument("--changed-vs", metavar="OLD_RESULTS.csv", help="also render only the tiles that differ from this results.csv")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="any other Params field")
    a = ap.parse_args()

    p = Params()
    over = [("min_bump_px", a.min_bump), ("min_dist_px", a.min_dist), ("min_width_px", a.min_width), ("tophat_rows", a.tophat_rows)]
    over += [tuple(s.split("=", 1)) for s in a.param]
    types = {f.name: f.type for f in dataclasses.fields(Params)}
    for k, v in over:
        if v is None:
            continue
        if k not in types:
            ap.error(f"unknown Params field {k}")
        setattr(p, k, v if not isinstance(v, str) else (int if "int" in str(types[k]) else float)(v))

    meta_path = os.path.join(a.cache, "meta.csv")
    if not os.path.isfile(meta_path):
        sys.exit(f"{meta_path} not found: run build_cache.py first")
    meta = list(csv.DictReader(open(meta_path, newline="")))
    meta.sort(key=lambda r: (int(r["index"]), int(r["part"]) if r["part"] else 0))
    if a.indices:
        want = {s.strip() for s in a.indices.split(",")}
        meta = [r for r in meta if r["index"] in want]
    if a.limit:
        keep = sorted({int(r["index"]) for r in meta})[:a.limit]
        meta = [r for r in meta if int(r["index"]) in set(keep)]
    if not meta:
        sys.exit("nothing to do")

    os.makedirs(a.qa, exist_ok=True)
    shutil.rmtree(os.path.join(a.qa, "sheets"), ignore_errors=True)       # no stale sheets from an earlier run
    os.makedirs(os.path.join(a.qa, "sheets"))
    tasks = [(n + 1, meta[i:i + PER_SHEET], p, a.cache, a.qa) for n, i in enumerate(range(0, len(meta), PER_SHEET))]
    print(f"{len(meta)} tiles ({sum(1 for r in meta if r['part'] == '')} no-match) -> {len(tasks)} sheets, jobs={a.jobs}", flush=True)

    t0, rows = time.time(), []
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for n, (no, rs) in enumerate(pool.imap(make_sheet, tasks), 1):
            rows += rs
            if n % 50 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} sheets  {time.time() - t0:.0f} s", flush=True)

    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["index"], []).append(r)
    frames = []
    for idx, rs in by_frame.items():
        fcls = frame_class([r["new"] for r in rs if r["part"] != ""])
        for r in rs:
            r["frame_new"] = fcls
        frames.append(dict(index=idx, n_parts=sum(1 for r in rs if r["part"] != ""), frame_new=fcls, old=rs[0]["old"]))
    write(os.path.join(a.qa, "results.csv"), RES_COLS, rows)
    write(os.path.join(a.qa, "frames_summary.csv"), ["index", "n_parts", "frame_new", "old"], frames)
    write(os.path.join(a.qa, "sheets_index.csv"), ["sheet", "tile_no", "index", "part", "new", "old"],
          [dict(sheet=f"sheet_{r['sheet']:04d}", tile_no=r["tile"], index=r["index"], part=r["part"], new=r["new"], old=r["old"])
           for r in rows])

    parts = [r for r in rows if r["part"] != ""]
    print(f"\n{len(frames)} frames, {len(parts)} parts, {sum(1 for f in frames if f['n_parts'] == 0)} no-match frames, "
          f"{len(tasks)} sheets -> {a.qa}")
    print("part verdicts: " + "  ".join(f"{k} {v}" for k, v in sorted(Counter(r["new"] for r in parts).items())))
    print("frame verdict (any NG part -> NG), new per old:")
    for o in sorted({f["old"] for f in frames}, key=lambda s: ("OK", "NG", "NA").index(s) if s in ("OK", "NG", "NA") else 9):
        sel = [f for f in frames if f["old"] == o]
        c = Counter(f["frame_new"] for f in sel)
        print(f"  old={o or '-':2s} ({len(sel):4d}) -> new OK {c['OK']}  NG {c['NG']}  NA {c['NA']}")
    if a.changed_vs:
        make_changed(a, rows, meta, p)
    for s in "LR":
        for k in ("dist", "bump"):
            v = np.array([float(r[f"{s}_{k}"]) for r in parts if r.get(f"{s}_{k}", "") != ""], float)
            if v.size:
                print(f"  {s}_{k}: p1/10/50/90/99 = " + " ".join(f"{np.percentile(v, q):.0f}" for q in (1, 10, 50, 90, 99)))


if __name__ == "__main__":
    main()
