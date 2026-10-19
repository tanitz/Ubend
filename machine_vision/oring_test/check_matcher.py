#!/usr/bin/env python3
"""Self-check of the native-matcher NaN workaround in test_oring_v2.Cropper.

For every sample frame it makes `--repeats` calls
  before: plain native call on the full frame (what the app / the old Cropper did): invalid = any
          returned match has a non-finite angle / corner / centre
  after : Cropper.crop_all(frame) (shifted retries): invalid = the result has a non-finite quad or fewer
          parts than the best answer seen for that frame (a part would be silently lost)
and prints the counts, the extra native calls the retries cost, and the spread of the accepted results.

  cd machine_vision
  ubend/bin/python oring_test/check_matcher.py                       # known bad frames + random ones, 20 x 10
  ubend/bin/python oring_test/check_matcher.py --frames 4125,2330 --repeats 20
Exit code 1 if the workaround still returns an invalid / lost part.
"""
import argparse, csv, os, random, sys
sys.dont_write_bytecode = True   # importing app/matching2.py must not rewrite the tracked .pyc files
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))
from test_oring_v2 import Cropper  # noqa: E402
from app import storage  # noqa: E402

KNOWN = [4125, 2330, 4016, 4390, 4118, 3266, 3652, 1783]   # NaN-prone, no-match (4118, 3266, 3652), 4 parts (1783)


def raw_call(cropper, frame):
    """(n_returned, n_invalid) of one plain native call."""
    n = min(cropper.m.match(frame), cropper.max_parts)
    bad = 0
    for i in range(max(n, 0)):
        r = cropper.m.results[i]
        v = [r.leftTopX, r.leftTopY, r.leftBottomX, r.leftBottomY, r.rightBottomX, r.rightBottomY,
             r.rightTopX, r.rightTopY, r.centerX, r.centerY, r.angle]
        bad += not (r.score > 0 and np.isfinite(v).all())
    return n, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(storage.report_dir() / "2026" / "09" / "report.csv"))
    ap.add_argument("--template", default=os.path.join(ROOT, "model", "model10.png"))
    ap.add_argument("--lib")
    ap.add_argument("--frames", help="comma separated frame indices (default: known bad frames + random ones)")
    ap.add_argument("--n", type=int, default=20, help="number of frames when --frames is not given")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = {r["index"].strip(): r for r in storage.read_report_rows(a.csv) if os.path.isfile(r["input_image"])}
    if a.frames:
        idx = [s.strip() for s in a.frames.split(",") if s.strip()]
    else:
        rest = sorted(set(rows) - {str(k) for k in KNOWN}, key=int)
        idx = [str(k) for k in KNOWN if str(k) in rows]
        idx += random.Random(a.seed).sample(rest, max(0, min(a.n - len(idx), len(rest))))
    idx = [i for i in idx if i in rows]
    cropper = Cropper(a.template, a.lib)

    print(f"{len(idx)} frames x {a.repeats} repeats", flush=True)
    before = {}
    for i in idx:
        frame = cv2.imread(rows[i]["input_image"], cv2.IMREAD_COLOR)
        calls = [raw_call(cropper, frame) for _ in range(a.repeats)]
        before[i] = (max(n for n, _ in calls), sum(b > 0 for _, b in calls), max(n - b for n, b in calls))

    tot = dict(b_calls=0, b_bad=0, a_calls=0, a_bad=0, native=0, native_bad=0)
    print(f"{'frame':>6} {'n':>2} | {'plain invalid':>13} | {'crop_all lost/invalid':>21} {'native calls':>12} {'NaN calls':>9} "
          f"{'dxy px':>7} {'dang deg':>8}")
    for i in idx:
        frame = cv2.imread(rows[i]["input_image"], cv2.IMREAD_COLOR)
        res = []
        for _ in range(a.repeats):
            parts = cropper.crop_all(frame)
            res.append((parts, dict(cropper.last)))
        expect = max(max(len(p) for p, _ in res), before[i][2])   # most valid parts any call returned
        lost = sum(len(p) < expect or any(not np.isfinite(q["quad"]).all() for q in p) for p, _ in res)
        native = sum(s["calls"] for _, s in res)
        nbad = sum(s["invalid_calls"] for _, s in res)
        # spread of the accepted detections of the same part (centre px, angle deg)
        dxy, dang = 0.0, 0.0
        ref = res[0][0]
        for p, _ in res[1:]:
            for q in p:
                for r in ref:
                    if np.hypot(q["center"][0] - r["center"][0], q["center"][1] - r["center"][1]) < 60:
                        dxy = max(dxy, float(np.hypot(q["center"][0] - r["center"][0], q["center"][1] - r["center"][1])))
                        dang = max(dang, abs((q["angle"] - r["angle"] + 180) % 360 - 180))
        print(f"{i:>6} {before[i][0]:>2} | {before[i][1]:>6}/{a.repeats:<6} | {lost:>12}/{a.repeats:<8} {native:>12} {nbad:>9} "
              f"{dxy:>7.2f} {dang:>8.2f}", flush=True)
        tot["b_calls"] += a.repeats; tot["b_bad"] += before[i][1]
        tot["a_calls"] += a.repeats; tot["a_bad"] += lost
        tot["native"] += native; tot["native_bad"] += nbad
    print(f"TOTAL before: {tot['b_bad']}/{tot['b_calls']} calls invalid ({100 * tot['b_bad'] / max(1, tot['b_calls']):.1f}%)")
    print(f"TOTAL after : {tot['a_bad']}/{tot['a_calls']} crop_all results invalid or lost parts; "
          f"{tot['native']} native calls for {tot['a_calls']} results ({tot['native'] / max(1, tot['a_calls']):.2f} per result), "
          f"{tot['native_bad']} of them returned NaN and were retried")
    sys.exit(1 if tot["a_bad"] else 0)


if __name__ == "__main__":
    main()
