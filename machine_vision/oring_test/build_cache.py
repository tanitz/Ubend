#!/usr/bin/env python3
"""Stage 1 of the QA pipeline: run the native matcher on every frame of a report and cache the result.

  cd machine_vision
  ubend/bin/python oring_test/build_cache.py --csv ~/UbendData/report/2026/09/report.csv --jobs 8
  nohup ubend/bin/python oring_test/build_cache.py --csv ~/UbendData/report/2026/09/report.csv > ~/UbendData/testing/logs/cache_build.log 2>&1 &

Writes to --cache (default <data_dir>/testing/cache):
  crops/<index>_p<k>.png   rectified crop (template size) of every part found in the frame, reading order
  preview/<index>.jpg      input frame downscaled to 900 px wide with the numbered green quads
  meta.csv                 index, part, score, angle, center_x, center_y, crop_path, input_path, old_result, note
                           (paths relative to --cache; a frame without a valid match has ONE row with an empty part
                           and note 'no match...')
  frames_stats.csv         per frame matcher statistics (native calls, NaN calls, unresolved parts, seconds)
Resumable: frames that already have rows in meta.csv are skipped (use --force to redo them).
Stage 2 is run_from_cache.py.
"""
import argparse, csv, fcntl, multiprocessing as mp, os, sys, time
sys.dont_write_bytecode = True   # importing app/matching2.py must not rewrite the tracked .pyc files
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from app import storage  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "app"))

META_COLS = ["index", "part", "score", "angle", "center_x", "center_y", "crop_path", "input_path", "old_result", "note"]
STAT_COLS = ["index", "n_parts", "calls", "invalid_calls", "bad_matches", "unresolved", "max_tries", "seconds"]
PREVIEW_W = 900

_cropper = None


def _init(template, lib):
    global _cropper
    cv2.setNumThreads(1)
    from test_oring_v2 import Cropper
    _cropper = Cropper(template, lib)


def preview_image(frame, parts):
    """Frame downscaled to PREVIEW_W with the numbered quads of the parts."""
    sc = PREVIEW_W / frame.shape[1]
    prev = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    for k, pt in enumerate(parts, 1):
        q = (pt["quad"] * sc).astype(np.int32)
        cv2.polylines(prev, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.putText(prev, f"#{k}", (int(q[:, 0].min()), max(18, int(q[:, 1].min()) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return prev


def _work(item):
    """One frame -> (index, meta rows, stats row). Runs in a worker process."""
    idx, path, old, cache = item
    t0 = time.time()
    base = dict(index=idx, part="", score="", angle="", center_x="", center_y="", crop_path="", input_path=path, old_result=old)
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        stats = dict(index=idx, n_parts=0, calls=0, invalid_calls=0, bad_matches=0, unresolved=0, max_tries=0, seconds=0)
        return idx, [dict(base, note="no match: unreadable image")], stats
    parts = _cropper.crop_all(frame)
    last = dict(_cropper.last)
    rows = []
    for k, pt in enumerate(parts, 1):
        rel = f"crops/{idx}_p{k}.png"
        cv2.imwrite(os.path.join(cache, rel), pt["crop"], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        note = []
        if pt["tries"] > 1:
            note.append(f"NaN retry {pt['tries']} (shift {pt['shift'][0]},{pt['shift'][1]})")
        if last["unresolved"]:
            note.append(f"{last['unresolved']} match(es) of this frame stayed NaN")
        rows.append(dict(base, part=k, score=f"{pt['score']:.4f}", angle=f"{pt['angle']:.3f}",
                         center_x=f"{pt['center'][0]:.1f}", center_y=f"{pt['center'][1]:.1f}", crop_path=rel,
                         note="; ".join(note)))
    if not parts:
        why = f"no match: {last['unresolved']} match(es) stayed NaN after {last['calls']} calls" if last["unresolved"] else "no match"
        rows.append(dict(base, note=why))
    cv2.imwrite(os.path.join(cache, "preview", f"{idx}.jpg"), preview_image(frame, parts), [cv2.IMWRITE_JPEG_QUALITY, 85])
    stats = dict(index=idx, n_parts=len(parts), calls=last["calls"], invalid_calls=last["invalid_calls"],
                 bad_matches=last["bad_matches"], unresolved=last["unresolved"],
                 max_tries=max((p["tries"] for p in parts), default=0), seconds=round(time.time() - t0, 2))
    return idx, rows, stats


def read_csv(path):
    """Rows of a csv written by this script; malformed (half written) lines are dropped."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        head = next(rd, None)
        if not head:
            return []
        return [dict(zip(head, r)) for r in rd if len(r) == len(head)]


def write_csv(path, cols, rows, mode="w"):
    new = mode == "w" or not os.path.isfile(path) or os.path.getsize(path) == 0
    if mode == "a" and not new:
        with open(path, "rb+") as f:                      # a killed run may have left a half line
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerows(rows)


def sort_key(r):
    return int(r["index"]), int(r["part"]) if str(r["part"]) != "" else 0


def summary(cache, selected=None):
    meta, stats = read_csv(os.path.join(cache, "meta.csv")), read_csv(os.path.join(cache, "frames_stats.csv"))
    if selected is not None:
        meta = [r for r in meta if r["index"] in selected]
        stats = [r for r in stats if r["index"] in selected]
    parts = [r for r in meta if r["part"] != ""]
    nomatch = [r for r in meta if r["part"] == ""]
    per = Counter(r["index"] for r in parts)
    calls = sum(int(r["calls"]) for r in stats)
    bad_calls = sum(int(r["invalid_calls"]) for r in stats)
    retried = sum(1 for r in stats if int(r["invalid_calls"]) > 0)
    unres = [r["index"] for r in stats if int(r["unresolved"]) > 0]
    tries = Counter(int(r["max_tries"]) for r in stats if int(r["n_parts"]) > 0)
    secs = sum(float(r["seconds"]) for r in stats)
    print(f"frames {len(stats)}  parts {len(parts)}  frames with 1/2/3/4+ parts: "
          + "/".join(str(sum(1 for v in per.values() if (v == n if n < 4 else v >= 4))) for n in (1, 2, 3, 4)))
    print(f"no-match frames {len(nomatch)} ({sum('NaN' in r['note'] for r in nomatch)} of them because every match stayed NaN)"
          + (f": {', '.join(r['index'] for r in nomatch[:15])}{' ...' if len(nomatch) > 15 else ''}" if nomatch else ""))
    print(f"native matcher calls {calls} ({calls / max(1, len(stats)):.2f} per frame), {bad_calls} returned a NaN match "
          f"-> {retried} frames needed a retry, {len(unres)} frames still had an unresolved NaN match"
          + (f" ({', '.join(unres[:10])})" if unres else ""))
    print(f"frames by number of calls until the last part was valid (max tries): "
          + ", ".join(f"{k}: {v}" for k, v in sorted(tries.items())) + f"   | worker time {secs / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(storage.report_dir() / "2026" / "09" / "report.csv"))
    ap.add_argument("--cache", default=str(storage.testing_dir("cache")))
    ap.add_argument("--template", default=os.path.join(ROOT, "model", "model10.png"))
    ap.add_argument("--lib")
    ap.add_argument("--jobs", type=int, default=8, help="worker processes (the native matcher uses ~1.4 cores each)")
    ap.add_argument("--limit", type=int, help="only the first N frames (benchmarks)")
    ap.add_argument("--indices", help="only these frame indices (comma separated)")
    ap.add_argument("--force", action="store_true", help="recompute frames that are already cached")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.cache, "crops"), exist_ok=True)
    os.makedirs(os.path.join(a.cache, "preview"), exist_ok=True)
    lock = open(os.path.join(a.cache, ".lock"), "w")          # two runs on one cache would append duplicate rows
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another build_cache.py is already writing to {a.cache}")
    items = [(r["index"].strip(), r["input_image"], r["result"].strip()) for r in storage.read_report_rows(a.csv)
             if os.path.isfile(r["input_image"])]
    items.sort(key=lambda t: int(t[0]))
    if a.indices:
        want = {s.strip() for s in a.indices.split(",")}
        items = [t for t in items if t[0] in want]
    if a.limit:
        items = items[:a.limit]
    selected = {t[0] for t in items}

    meta_path, stats_path = os.path.join(a.cache, "meta.csv"), os.path.join(a.cache, "frames_stats.csv")
    old_meta = read_csv(meta_path)
    if a.force:
        keep = [r for r in old_meta if r["index"] not in selected]
        write_csv(meta_path, META_COLS, keep)
        write_csv(stats_path, STAT_COLS, [r for r in read_csv(stats_path) if r["index"] not in selected])
        old_meta = keep
    have = {}
    for r in old_meta:
        have.setdefault(r["index"], []).append(r)
    have_stats = {r["index"] for r in read_csv(stats_path)}
    done = {i for i, rs in have.items()
            if i in have_stats and os.path.isfile(os.path.join(a.cache, "preview", f"{i}.jpg"))
            and all(r["part"] == "" or os.path.isfile(os.path.join(a.cache, r["crop_path"])) for r in rs)}
    todo = [(i, p, o, a.cache) for i, p, o in items if i not in done]
    print(f"{len(items)} frames selected, {len(items) - len(todo)} already cached, {len(todo)} to do, jobs={a.jobs}", flush=True)

    t0, n = time.time(), 0
    if todo:
        # frames whose rows are incomplete (crash) are redone: drop their old rows first
        redo = {t[0] for t in todo} & set(have)
        if redo:
            write_csv(meta_path, META_COLS, [r for r in old_meta if r["index"] not in redo])
            write_csv(stats_path, STAT_COLS, [r for r in read_csv(stats_path) if r["index"] not in redo])
        pool = mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(a.template, a.lib))
        try:
            for idx, rows, stats in pool.imap_unordered(_work, todo, chunksize=1):
                write_csv(meta_path, META_COLS, rows, "a")
                write_csv(stats_path, STAT_COLS, [stats], "a")
                n += 1
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)} frames  {el / 60:.1f} min  ({n / el:.2f} frames/s, eta {(len(todo) - n) / (n / el) / 60:.1f} min)",
                          flush=True)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            print("interrupted: run the same command again to resume", flush=True)
            sys.exit(130)
        finally:
            pool.join()
        print(f"processed {n} frames in {(time.time() - t0) / 60:.1f} min", flush=True)

    for path, cols in ((meta_path, META_COLS), (stats_path, STAT_COLS)):         # sorted, atomic rewrite
        rows = sorted(read_csv(path), key=sort_key if path == meta_path else (lambda r: int(r["index"])))
        write_csv(path + ".tmp", cols, rows)
        os.replace(path + ".tmp", path)
    print("--- summary (selected frames) ---")
    summary(a.cache, selected)


if __name__ == "__main__":
    main()
