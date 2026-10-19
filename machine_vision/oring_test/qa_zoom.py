#!/usr/bin/env python3
"""Zoom helper for reviewers: raw crop | v2 overlay | foreground mask of ONE cached part, plus its numbers.

  cd machine_vision
  ubend/bin/python oring_test/qa_zoom.py 1783 2              # part 2 of frame 1783 -> prints numbers + the PNG path
  ubend/bin/python oring_test/qa_zoom.py 4118                # frame without a part: prints the full-frame preview path
  ubend/bin/python oring_test/qa_zoom.py 1783 2 --scale 3 --out /some/dir

Panels (left to right): RAW crop, V2 overlay (marker line = detected ring, blue line = to the leg tip),
FOREGROUND MASK that the detector used (white = copper part). Uses the current oring_v2.py with default Params.
"""
import argparse, csv, os, sys
sys.dont_write_bytecode = True
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oring_v2  # noqa: E402
from app import storage  # noqa: E402


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("index")
    ap.add_argument("part", nargs="?", default="1")
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--cache", default=str(storage.testing_dir("cache")))
    ap.add_argument("--out", default=str(storage.testing_dir("results", "qa_zoom_tmp")))
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(os.path.join(a.cache, "meta.csv"), newline="")) if r["index"] == str(a.index)]
    if not rows:
        sys.exit(f"frame {a.index} not in cache")
    prev = os.path.join(a.cache, "preview", f"{a.index}.jpg")
    print(f"frame {a.index}: {sum(1 for r in rows if r['part'])} part(s) in the matcher cache; input frame: {rows[0]['input_path']}")
    print(f"full-frame preview (900 px wide, numbered green quads): {prev}")
    row = next((r for r in rows if r["part"] == str(a.part)), None)
    if row is None:
        print("no such part (no-match frame or wrong part number)")
        return
    crop = cv2.imread(os.path.join(a.cache, row["crop_path"]), cv2.IMREAD_COLOR)
    vis, cls, info = oring_v2.find_oring_v2(crop)
    mask = oring_v2._foreground(crop, oring_v2.Params())
    s = a.scale
    up = lambda im: cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if im.ndim == 3 and im is not mask else cv2.INTER_NEAREST)
    raw = label(up(crop), f"RAW  #{a.index} p{a.part}")
    ov = label(up(vis), f"V2 OVERLAY  new={cls}")
    mk = label(cv2.cvtColor(up(mask), cv2.COLOR_GRAY2BGR), "FOREGROUND MASK")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.index}_p{a.part}.png")
    cv2.imwrite(path, np.hstack([raw, ov, mk]))
    print(f"zoom image: {path}")
    print(f"matcher: score={row['score']} angle={row['angle']} old_system={row['old_result']}   v2 verdict={cls}  status={info.get('status')}")
    for side in "LR":
        d = info.get(side, {})
        if "dist" in d:
            print(f"  {side}: ring y={d['y']} x={d['x']} width={d['width']:.0f} tube={d['tube_width']:.0f} dist_to_tip={d['dist']:.0f} "
                  f"bump={d['bump']:.1f} thick={d['thickness']} tilt={d['tilt_raw']:.1f} cut={d.get('cut')}  {d.get('reason', '')}")
        else:
            print(f"  {side}: {d.get('reason', info.get('reason', '-'))}")


if __name__ == "__main__":
    main()
