"""O-ring locator v2 - used by the app (app/ui/home_page.py) and by the offline tools in oring_test/.

Input : rectified crop of the U-bend (what rotation_matching2() returns, BGR).
Output: ring position (x, y), ring width and distance-to-leg-tip for the LEFT
        and RIGHT leg, plus an annotated image.

What changed vs. HomePage.find_oring (see also README in the reply):
  1. The ring itself is located. Old code boxed the whole leg silhouette
     (S-channel mask -> Canny -> contours) and never separated the ring.
     Here the ring is the thin flange that makes the leg *wider* for a few
     rows: width-profile of each leg -> white top-hat -> strongest peak.
  2. Distance is measured to the real leg tip (last foreground row of that leg),
     not to the bottom edge of the crop, so it no longer depends on the crop.
  3. Foreground = copper-colour seed (hue/saturation) + pale flange pixels near it, instead of
     fixed S thresholds (object_thresh/hole_thresh): independent of the background colour, so it
     works on the grey/textured wall; keeps the pale flange and the inner tube (separate blob).
  4. Left/right are split at the gap between the legs (x of the emptiest column
     under the arch), not at the geometric centre of the crop.
  5. Every parameter comes from one `Params` dataclass (no silent overrides of
     config.ini values by config_manager, no cv2.imshow / inpaint / prints).
  6. Returns measurements, a confidence-like `bump` value and a reason string,
     so thresholds can be calibrated instead of guessed.
  7. Only the part in the middle of the crop is measured (neighbouring parts that intrude from the crop
     border used to corrupt the width profile of a leg).
  8. The ring is searched up to `ring_win_w` tube widths above the leg tip (scale independent) in a band
     around the leg axis; a leaning leg (axis tilt > min_tilt_deg) is also measured after rotating it upright
     about its tip and that result is used when it finds a clearly stronger ring. A leg that is cut by the
     crop border or leans more than na_tilt_deg (rotated / 3D-tilted part) gives NA instead of a false NG.
  Verdict: a leg is OK / NG / NA, the part is NG if any leg is NG, else NA if any leg is NA, else OK.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class Params:
    blur: int = 3                 # gaussian kernel on the distance map before Otsu (odd)
    min_fg: int = 40              # Otsu threshold on the (4x Lab distance) map is never allowed below this
    seed_s: int = 60              # copper seed: min saturation
    seed_hmax: int = 30           # copper seed: max hue (OpenCV 0-179; also accepts >=170 for red-brown)
    seed_v: int = 25              # copper seed: min value (low: the dark outline of the flange must stay connected)
    halo: int = 3                 # px around the seed where pale flange pixels may be added
    plain_bg_noise: float = 12.0  # border noise (Lab distance x4, p90) below which the halo step is used
    min_blob_frac: float = 0.08   # keep blobs >= this fraction of the biggest one (inner tube is a separate blob)
    low_fg: int = 16              # weak threshold for hysteresis (pale flange)
    tip_guard: int = 12           # ignore bumps in the last N rows (open tube end flares)
    leg_rows_frac: float = 0.55   # search legs only in the bottom part of the foreground (fraction of its height)
    tophat_rows: int = 25         # must be taller than the ring thickness, shorter than the arch curve
    min_bump_px: float = 4.0      # ring must protrude at least this much (px) over the tube width
    max_ring_rows: int = 30       # reject "bumps" thicker than this (they are the arch, not a flange)
    min_dist_px: float = 40.0     # ring must be at least this far above the leg tip   (old: h_oring)
    min_width_px: float = 0.0     # ring width must exceed this (old: w_oring=72); 0 = off, use bump instead
    ring_win_w: float = 1.5       # the ring is searched up to this many tube widths above the leg tip
    band_w: float = 0.9           # only pixels within +-(band_w * tube width + 6 px) of the leg axis are measured
    min_tilt_deg: float = 4.0     # leg axis tilt (from the row centres above the tip) above which an upright version is tried
    max_tilt_deg: float = 30.0    # larger estimates are not trusted (the leg is measured as it is)
    tilt_win_w: float = 1.0       # the axis is fitted on the rows up to this many tube widths above the tip
    tilt_gain_px: float = 3.0     # the upright version must beat the as-is one by this bump (px) to be used
    na_tilt_deg: float = 15.0     # a leg leaning more than this (rotated / 3D-tilted part) that fails is NA, not NG
    max_dist_px: float = 75.0     # the ring is searched at most this far above the leg tip (real ones: 45-66 px; farther = arch)
    max_dist_ok_px: float = float('inf')   # a ring farther than this from the tip is NG (app settings: nominal + tolerance); inf = only the search window limits it
    draw_labels: bool = True      # write 'L d:.. w:.. b:..' next to every ring (the verdict title is always drawn)
    min_tube_w_px: float = 35.0   # tube widths outside min..max (typical 54-73) mean a neighbour is attached or the leg is cut -> NA
    max_tube_w_px: float = 100.0
    max_ring_ratio: float = 1.8   # ring wider than this x tube width (typical 1.13-1.47) -> NA (neighbour attached)


def _keep_blobs(mask: np.ndarray, p: Params) -> np.ndarray:
    """Keep every sizeable blob (the part is often two separate concentric tubes with a white gap,
    so "biggest blob only" would silently drop the inner tube); drops dust / border shadows."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= max(150, p.min_blob_frac * areas.max()))
    return (np.isin(lab, keep) * 255).astype(np.uint8)


def _main_part(mask: np.ndarray) -> np.ndarray:
    """Keep the part in the middle of the crop: the blob with most pixels in the central window, the blobs whose centre
    lies inside its bounding box (inner tube) and big blobs centred in the window. Drops neighbouring parts that
    intrude from the crop border (they would corrupt the width profile of a leg)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    h, w = mask.shape
    x0, x1, y0, y1 = int(0.15 * w), int(0.85 * w), int(0.1 * h), int(0.9 * h)
    inside = np.bincount(lab[y0:y1, x0:x1].ravel(), minlength=n)[1:]
    area = st[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(inside if inside.max() > 0 else area))
    x, y, bw, bh = st[main, :4]
    keep = [main]
    for i in range(1, n):
        cx, cy = cen[i]
        in_box = x - 6 <= cx <= x + bw + 6 and y - 6 <= cy <= y + bh + 6
        centred = x0 <= cx <= x1 and y0 <= cy <= y1 and st[i, cv2.CC_STAT_AREA] >= 0.2 * st[main, cv2.CC_STAT_AREA]
        if i != main and (in_box or centred):
            keep.append(i)
    return (np.isin(lab, keep) * 255).astype(np.uint8)


def _lab_distance(img: np.ndarray, p: Params):
    """4x colour distance (Lab) from the background colour sampled on the crop border + border noise level."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = 4
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    d = np.linalg.norm(lab - bg, axis=2) * 4.0
    d = cv2.GaussianBlur(np.clip(d, 0, 255).astype(np.uint8), (p.blur | 1, p.blur | 1), 0)
    bd = np.concatenate([d[:b].ravel(), d[-b:].ravel(), d[:, :b].ravel(), d[:, -b:].ravel()])
    return d, float(np.percentile(bd, 90))


def _foreground(img: np.ndarray, p: Params) -> np.ndarray:
    """Foreground of the copper part.
    1) Seed = copper colour (orange hue, saturated). Independent of the background, so it also works on
       the grey / textured wall where a "distance from background colour" mask fills the whole crop.
    2) On a plain (low-noise) background, add pale pixels (the flange edge is nearly white) that lie within
       `halo` px of the seed, using the Lab distance from the background.
    3) If there is no copper-coloured seed at all, fall back to Otsu on the Lab distance."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    seed = ((s >= p.seed_s) & ((h <= p.seed_hmax) | (h >= 170)) & (v >= p.seed_v)).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    d, noise = _lab_distance(img, p)
    if int((seed > 0).sum()) >= 300:
        mask = seed
        if noise <= p.plain_bg_noise:
            k = 2 * p.halo + 1
            near = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            loose = (d > p.low_fg).astype(np.uint8) * 255
            mask = cv2.bitwise_or(seed, cv2.bitwise_and(loose, near))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return _main_part(_keep_blobs(mask, p))
    low_fg = min(40, max(p.low_fg, int(noise) + 6))
    min_fg = max(p.min_fg, low_fg + 24)
    t, mask = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if t < min_fg:
        _, mask = cv2.threshold(d, min_fg, 255, cv2.THRESH_BINARY)
    return _main_part(_keep_blobs(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), p))


def _split_x(mask: np.ndarray, y_top: int, y_bot: int) -> int:
    """x of the gap between the two legs, measured in the lower part of the shape."""
    cols = (mask[y_top:y_bot] > 0).sum(axis=0).astype(np.float32)
    xs = np.flatnonzero(cols)
    if xs.size == 0:
        return mask.shape[1] // 2
    x0, x1 = xs[0], xs[-1]
    span = x1 - x0
    lo, hi = int(x0 + 0.3 * span), int(x0 + 0.7 * span)
    seg = cv2.GaussianBlur(cols[None, lo:hi + 1], (0, 0), 5)[0]
    return lo + int(np.argmin(seg))


def _extent(sub: np.ndarray):
    """Per row: has pixels, first column, last column, width of a boolean array."""
    has = sub.any(axis=1)
    xmin = np.where(has, sub.argmax(axis=1), 0)
    xmax = np.where(has, sub.shape[1] - 1 - sub[:, ::-1].argmax(axis=1), 0)
    return has, xmin, xmax, np.where(has, xmax - xmin + 1, 0).astype(np.float32)


def _ring_on_axis(m: np.ndarray, xc: float, slope: float, tip_y: int, w0: float, p: Params):
    """Width profile of one leg inside a band around its axis x = xc + slope * (y - tip_y), white top-hat -> ring.
    Returns (dict in the pixel frame of m, None) or (None, reason)."""
    h, w = m.shape
    rows = np.arange(max(0, tip_y - int(2.6 * w0) - p.tophat_rows), min(h, tip_y + 30))
    if rows.size < 30:
        return None, "leg too close to the crop border"
    cx = xc + slope * (rows - tip_y)
    near = np.abs(np.arange(w)[None, :] - cx[:, None]) <= p.band_w * w0 + 6
    has, xmin, xmax, width = _extent(near & (m[rows] > 0))
    solid = width > 0.5 * w0
    if not solid.any():
        return None, "no solid rows"
    tip = int(np.flatnonzero(solid)[-1])                      # leg tip (index into rows)
    # smooth width profile, then white top-hat -> thin flange stands out
    prof = cv2.GaussianBlur(width[None, :tip + 1], (0, 0), 1.2)[0]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, p.tophat_rows))
    th = cv2.morphologyEx(prof[:, None], cv2.MORPH_TOPHAT, k)[:, 0]
    # ignore the tip itself (open end can flare) and everything farther than ring_win_w tube widths / max_dist_px from it
    cand = th.copy()
    cand[:max(0, tip - int(min(p.ring_win_w * w0, p.max_dist_px)))] = 0
    cand[max(0, tip - p.tip_guard):] = 0
    if cand.max() <= 0:
        return None, "no bump"
    peak = int(np.argmax(cand))
    bump = float(cand[peak])
    a = peak
    while a > 0 and th[a - 1] >= 0.5 * bump:
        a -= 1
    b = peak
    while b < tip and th[b + 1] >= 0.5 * bump:
        b += 1
    ring = (a + b) // 2
    seg = slice(a, b + 1)
    xl, xr = int(xmin[seg].min()), int(xmax[seg].max())
    win = slice(max(0, tip - int(min(p.ring_win_w * w0, p.max_dist_px))), tip + 1)
    return dict(
        pts=np.array([[xl, rows[ring]], [xr, rows[ring]], [0.5 * (xl + xr), rows[ring]], [xc + slope * (rows[tip] - tip_y), rows[tip]]], np.float32),
        width=float(xr - xl + 1), bump=bump, thickness=int(b - a + 1), dist=float(tip - ring),
        cut=bool(rows[tip] >= h - 2 or (xmin[win][has[win]] <= 0).any() or (xmax[win][has[win]] >= w - 1).any())), None


def _analyse_leg(mask: np.ndarray, y_top: int, y_bot: int, p: Params):
    """mask: foreground restricted to one half (full size). Finds the leg tip and tube width, estimates the leg axis
    from the row centres above the tip, measures the ring inside a band around that axis and, if the leg leans,
    also after rotating it upright about the tip (kept only when that finds a clearly stronger ring).
    Returns (dict, None) or (None, reason); points are in crop coordinates."""
    sub = mask[y_top:y_bot] > 0
    has, xmin, xmax, width = _extent(sub)
    if has.sum() < 20:
        return None, "no leg pixels"
    ref_w = float(np.median(width[has]))
    if ref_w <= 0:
        return None, "zero width"
    solid = width > 0.5 * ref_w
    if not solid.any():
        return None, "no solid rows"
    tip_i = int(np.flatnonzero(solid)[-1])                    # last solid row = leg tip (rough)
    seg0 = slice(max(0, tip_i - 40), max(1, tip_i - 9))
    ws = width[seg0][solid[seg0]]
    w0 = float(np.median(ws)) if ws.size else ref_w           # tube width below the flange
    if not p.min_tube_w_px <= w0 <= p.max_tube_w_px:
        return None, f"tube width {w0:.0f} px implausible (neighbour attached or leg cut?)"
    tip_y, xc, slope, tilt, tilt_raw = y_top + tip_i, float(0.5 * (xmin[tip_i] + xmax[tip_i])), 0.0, 0.0, 0.0
    r = np.arange(max(0, tip_i - int(p.tilt_win_w * w0)), max(0, tip_i - 5))
    r = r[has[r]] if r.size else r
    # a leg cut by the left / right crop border has a biased centre line: do not trust its axis
    if r.size >= 15 and not ((xmin[r] <= 0).any() or (xmax[r] >= mask.shape[1] - 1).any()):
        c = 0.5 * (xmin[r] + xmax[r])
        ok = np.ones(r.size, bool)
        for _ in range(3):
            sl, ic = np.polyfit(r[ok], c[ok], 1)
            res = np.abs(c - (ic + sl * r))
            ok = res <= max(2.0, 2.5 * float(res[ok].std()))
            if ok.sum() < 10:
                break
        tilt_raw = float(np.degrees(np.arctan(sl)))
        if abs(tilt_raw) <= p.max_tilt_deg:
            slope, tilt, xc = float(sl), tilt_raw, float(ic + sl * tip_i)
    res, why = _ring_on_axis(mask, xc, slope, tip_y, w0, p)
    if abs(tilt) >= p.min_tilt_deg:
        M = cv2.getRotationMatrix2D((xc, float(tip_y)), -tilt, 1.0)
        up = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
        res2, _ = _ring_on_axis(up, xc, 0.0, tip_y, w0, p)
        if res2 is not None and (res is None or res2["bump"] >= res["bump"] + p.tilt_gain_px):
            res2["pts"] = cv2.transform(res2["pts"][None], cv2.invertAffineTransform(M))[0]
            res, why = res2, None
            res["upright"] = True
    if res is None:
        return None, why
    pts = np.rint(res.pop("pts")).astype(int).tolist()
    res.update(y=pts[2][1], x=pts[2][0], x_left=pts[0][0], x_right=pts[1][0],
               p_left=tuple(pts[0]), p_right=tuple(pts[1]), p_tip=tuple(pts[3]), tip_y=pts[3][1],
               tube_width=w0, tilt=tilt, tilt_raw=tilt_raw, upright=res.get("upright", False))
    return res, None


def find_oring_v2(img: np.ndarray, params: Params | None = None):
    """Returns (annotated_bgr, classification, info). classification in OK/NG/NA."""
    p = params or Params()
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return np.zeros((100, 100, 3), np.uint8), "NA", {"reason": "empty image"}
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = _foreground(img, p)
    ys, xs = np.nonzero(mask)
    vis = img.copy()
    if ys.size < 200:
        cv2.putText(vis, "NA: no object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return vis, "NA", {"reason": "no object"}
    y0, y1 = int(ys.min()), int(ys.max())
    y_top = y0 + int((1.0 - p.leg_rows_frac) * (y1 - y0))   # only the legs (below the arch crest)
    y_bot = y1 + 1
    xs_split = _split_x(mask, y_top, y_bot)
    halves = {"L": (0, xs_split), "R": (xs_split, mask.shape[1])}
    info, status = {"split_x": xs_split}, {}
    for side, (xa, xb) in halves.items():
        m = np.zeros_like(mask)
        m[:, xa:xb] = mask[:, xa:xb]
        res, why = _analyse_leg(m, y_top, y_bot, p)
        if res is None:
            status[side] = "NA"
            info[side] = {"reason": why}
            continue
        ok = (res["bump"] >= p.min_bump_px and res["thickness"] <= p.max_ring_rows
              and p.min_dist_px <= res["dist"] <= p.max_dist_ok_px and res["width"] > p.min_width_px)
        leaning = abs(res["tilt_raw"]) > p.na_tilt_deg       # rotated / foreshortened leg: a failing check proves nothing
        fat = res["width"] > p.max_ring_ratio * res["tube_width"]
        status[side] = "NA" if (ok and fat) else ("OK" if ok else ("NA" if (res["cut"] or leaning) else "NG"))
        if ok and fat:
            res["reason"] = f"ring {res['width']:.0f} px = {res['width'] / res['tube_width']:.1f} x tube width (neighbour attached?)"
        if not ok:
            res["reason"] = ", ".join(r for r, bad in (
                (f"bump {res['bump']:.1f}<{p.min_bump_px}", res["bump"] < p.min_bump_px),
                (f"thick {res['thickness']}>{p.max_ring_rows}", res["thickness"] > p.max_ring_rows),
                (f"dist {res['dist']:.0f}<{p.min_dist_px}", res["dist"] < p.min_dist_px),
                (f"dist {res['dist']:.0f}>{p.max_dist_ok_px:g}", res["dist"] > p.max_dist_ok_px),
                (f"width {res['width']:.0f}<={p.min_width_px}", res["width"] <= p.min_width_px)) if bad)
            if res["cut"]:
                res["reason"] = "leg cut by crop border: " + res["reason"]
            elif leaning:
                res["reason"] = f"leg leans {res['tilt_raw']:.0f} deg: " + res["reason"]
        info[side] = res
        col = {"OK": (0, 200, 0), "NG": (0, 0, 255), "NA": (0, 200, 255)}[status[side]]
        weak = res["bump"] < p.min_bump_px                   # no flange found: the marked row is just the best of nothing
        cv2.line(vis, res["p_left"], res["p_right"], col, 1 if weak else 2)
        cv2.circle(vis, (res["x"], res["y"]), 4, col, -1)
        cv2.line(vis, (res["x"], res["y"]), res["p_tip"], (255, 160, 0), 1)
        if p.draw_labels or weak:
            cv2.putText(vis, f"{side} no ring b:{res['bump']:.0f}" if weak else
                        f"{side} d:{res['dist']:.0f} w:{res['width']:.0f} b:{res['bump']:.0f}",
                        (max(0, res["x_left"] - 10), max(12, res["y"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.line(vis, (xs_split, y_top), (xs_split, y_bot), (200, 200, 200), 1)
    s = list(status.values())
    cls = "NG" if "NG" in s else ("NA" if "NA" in s else "OK")
    col = {"OK": (0, 200, 0), "NG": (0, 0, 255), "NA": (0, 200, 255)}[cls]
    cv2.putText(vis, f"oring1 {status['L'].lower()} | oring2 {status['R'].lower()}", (50, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    info["status"] = status
    info["params"] = asdict(p)
    return vis, cls, info


def make_montage(images, cols=None, cell_h=None):
    """Tile several BGR images (one per part) into one picture for the HOME page / report. One image is returned as is."""
    images = [im for im in images if im is not None]
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    cols = cols or (2 if len(images) <= 4 else 3)
    cell_h = cell_h or max(im.shape[0] for im in images)
    ims = [cv2.resize(im, (max(1, int(im.shape[1] * cell_h / im.shape[0])), cell_h)) for im in images]
    w = max(im.shape[1] for im in ims)
    ims = [cv2.copyMakeBorder(im, 0, 0, 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for im in ims]
    while len(ims) % cols:
        ims.append(np.full_like(ims[0], 255))
    return np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])
