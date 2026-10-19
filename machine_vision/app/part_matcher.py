"""Finds every U-bend in a camera frame (native template matcher) and returns one rectified crop per part.

Replaces HomePage.rotation_matching2, which took only the best match and trusted the native result blindly.

The native matcher (native/libtemplatematching_ctype.so via app/matching2.py) intermittently returns NaN
angle / corners / centre for the SAME image (its sub-pixel step reads uninitialised memory, so the result depends
on the heap state). Warping with such a result gives a blank crop. Every result is therefore checked for
finiteness; on a bad one the match is repeated on the frame cropped by a few (dx, dy) offsets (this changes the
buffer sizes and the pixel grid) and the offset is added back, so coordinates stay in the original frame.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from app import matching2

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
NATIVE_DIR = os.path.join(ROOT_DIR, "native")
LIB_NAME = "libtemplatematching_ctype.so"
LEGACY_LIB_DIR = "/home/snc/Documents/DEMO/opencv_matching/build/matcher"   # only a fallback for this PC

MAX_PARTS = 8
SCORE_THRESHOLD = 0.55

# (dx, dy) by which the frame is cropped (top-left) before it is fed to the native matcher on a retry; the last
# ones are big because a few frames stay NaN for every shift below ~16 px
SHIFTS = [(0, 0), (5, 3), (2, 7), (9, 1), (7, 8), (3, 11), (12, 4), (1, 5), (20, 20), (33, 17), (50, 40), (100, 60)]


def find_native_lib() -> str:
    """Path of libtemplatematching_ctype.so: $TEMPLATE_MATCHING_SO, then native/, then the old DEMO build."""
    candidates = [os.environ.get("TEMPLATE_MATCHING_SO", ""),
                  os.path.join(NATIVE_DIR, LIB_NAME),
                  os.path.join(LEGACY_LIB_DIR, LIB_NAME)]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(f"{LIB_NAME} not found; tried: {[c for c in candidates if c]}")


class PartMatcher:
    """One instance per template. `crop_all(frame)` -> list of dicts(crop, quad, score, angle, center, tries, shift)
    in reading order (top-to-bottom, left-to-right); `quad` is in frame coordinates (lt, lb, rb, rt).
    `last` holds the statistics of the latest call (calls, invalid_calls, bad_matches, unresolved, parts)."""

    def __init__(self, template, lib: str | None = None, score: float = SCORE_THRESHOLD, max_parts: int = MAX_PARTS):
        tpl = template if template.ndim == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.max_parts = max_parts
        self.m = matching2.Matcher(lib or find_native_lib(), maxCount=max_parts, scoreThreshold=score,
                                   iouThreshold=0.4, angle=360, minArea=64)
        self.m.set_template(tpl)
        self.th, self.tw = tpl.shape[:2]
        self.dst = np.array([[0, 0], [0, self.th - 1], [self.tw - 1, self.th - 1], [self.tw - 1, 0]], np.float32)
        self.last: dict = {}

    @property
    def template_size(self):
        """(width, height) of the template = size of every crop."""
        return self.tw, self.th

    @staticmethod
    def _offset(k):
        return SHIFTS[k] if k < len(SHIFTS) else ((7 * k) % 15 + 1, (5 * k) % 13 + 1)

    def _match_once(self, gray, off):
        """One native call on gray[dy:, dx:] -> (n_returned, valid [(score, angle, quad, centre)] in frame coordinates,
        n_invalid). Values are copied out: results[i] is a view of the buffer that the next call overwrites."""
        dx, dy = off
        g = np.ascontiguousarray(gray[dy:, dx:]) if (dx or dy) else gray
        n = self.m.match(g)
        shift = np.array([dx, dy], np.float32)
        good, bad = [], 0
        for i in range(max(0, min(n, self.max_parts))):
            r = self.m.results[i]
            c = np.array([[r.leftTopX, r.leftTopY], [r.leftBottomX, r.leftBottomY],
                          [r.rightBottomX, r.rightBottomY], [r.rightTopX, r.rightTopY]], np.float32)
            ctr = np.array([r.centerX, r.centerY], np.float32)
            if not (r.score > 0 and np.isfinite(c).all() and np.isfinite(ctr).all() and np.isfinite(r.angle)):
                bad += 1
                continue
            good.append((float(r.score), float(r.angle), c + shift, ctr + shift))
        return n, good, bad

    def crop_all(self, frame, retries=len(SHIFTS), confirm_empty=1):
        """Calls the matcher on the frame; while a call returns an invalid (non-finite) match it is repeated on a
        shifted copy (up to `retries` calls), valid parts of all calls are merged (the first valid detection of a part
        wins). An empty answer is re-checked on `confirm_empty` shifted copies before it counts as 'no match'."""
        if frame is None or frame.size == 0:
            self.last = dict(calls=0, invalid_calls=0, bad_matches=0, unresolved=0, parts=0)
            return []
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, calls, invalid_calls, bad_seen, n_last, empty_checks = [], 0, 0, 0, 0, 0
        for k in range(retries):
            off = self._offset(k)
            n, good, bad = self._match_once(gray, off)
            calls += 1
            n_last = n
            bad_seen += bad
            invalid_calls += (bad > 0 or n < 0)
            for score_, angle_, c, ctr in good:
                if all(np.hypot(*(ctr - f["ctr"])) > 0.25 * self.th for f in found):
                    found.append(dict(score=score_, angle=angle_, c=c, ctr=ctr, tries=k + 1, shift=off))
            if n == 0 and not found and empty_checks < confirm_empty:
                empty_checks += 1
                continue
            if bad == 0 and n >= 0:
                break
        parts = []
        for f in found:
            M = cv2.getPerspectiveTransform(f["c"], self.dst)
            crop = cv2.warpPerspective(frame, M, (self.tw, self.th), borderValue=(255, 255, 255))
            parts.append(dict(crop=crop, quad=f["c"], score=f["score"], angle=f["angle"],
                              center=(float(f["ctr"][0]), float(f["ctr"][1])), tries=f["tries"], shift=f["shift"]))
        parts.sort(key=lambda p: (round(p["center"][1] / 400), p["center"][0]))
        self.last = dict(calls=calls, invalid_calls=int(invalid_calls), bad_matches=bad_seen,
                         unresolved=max(0, min(n_last, self.max_parts) - len(parts)), parts=len(parts))
        return parts

    find_parts = crop_all

    def best(self, frame):
        """The single best part (highest score) or None."""
        parts = self.crop_all(frame)
        return max(parts, key=lambda q: q["score"]) if parts else None
