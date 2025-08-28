import cv2
import numpy as np
import math

from concurrent.futures import ThreadPoolExecutor

D2R = math.pi / 180.0

def rotate_image(image, angle_deg, border_value=255):
    center = (image.shape[1] / 2, image.shape[0] / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int(image.shape[0] * sin + image.shape[1] * cos)
    new_h = int(image.shape[0] * cos + image.shape[1] * sin)
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=border_value)
    return rotated, rot_mat

def invert_affine_transform(mat):
    return cv2.invertAffineTransform(mat)

def apply_affine_transform(pt, mat_inv):
    pt = np.array([[pt]], dtype=np.float32)
    pt_transformed = cv2.transform(pt, mat_inv)
    return tuple(pt_transformed[0][0])

def draw_rotated_match(image, center, w, h, angle_deg, color=(0, 255, 0), label=None):
    cx, cy = float(center[0]), float(center[1])
    rect = ((cx, cy), (float(w), float(h)), float(angle_deg))
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.polylines(image, [box], isClosed=True, color=color, thickness=2)
    cv2.drawMarker(image, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 20, 2)
    if label:
        cv2.putText(image, label, (int(cx + 5), int(cy - 5)), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2)
    return image

def rotated_rect_iou(rect1, rect2):
    inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if inter_type == 0 or inter_pts is None or len(inter_pts) < 3:
        return 0.0
    inter_area = cv2.contourArea(inter_pts)
    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def non_maximum_suppression_iou(matches, iou_thresh=0.3, template_size=(100, 100)):
    w, h = template_size
    matches = sorted(matches, key=lambda x: -x['score'])
    suppressed = []
    taken = [False] * len(matches)
    for i in range(len(matches)):
        if taken[i]:
            continue
        m1 = matches[i]
        rect1 = (tuple(m1['position']), (w, h), m1['angle'])
        suppressed.append(m1)
        for j in range(i + 1, len(matches)):
            if taken[j]:
                continue
            m2 = matches[j]
            rect2 = (tuple(m2['position']), (w, h), m2['angle'])
            iou = rotated_rect_iou(rect1, rect2)
            if iou > iou_thresh:
                taken[j] = True
    return suppressed

def match_template_rotated_fast(src, template, angle_range=(-15, 15), angle_step=5, candidates_per_angle=1, min_score=0.5, scale=0.5):
    results = []
    src_small = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    template_small = cv2.resize(template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    th, tw = template_small.shape[:2]
    for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
        rotated, matR = rotate_image(src_small, angle)
        # Skip if template is larger than rotated image
        if template_small.shape[0] > rotated.shape[0] or template_small.shape[1] > rotated.shape[1]:
            continue
        result = cv2.matchTemplate(rotated, template_small, cv2.TM_CCOEFF_NORMED)
        for _ in range(candidates_per_angle):
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < min_score:
                break
            center_rotated = (max_loc[0] + tw / 2, max_loc[1] + th / 2)
            inv_matR = invert_affine_transform(matR)
            mapped_center = apply_affine_transform(center_rotated, inv_matR)
            mapped_center = (mapped_center[0] / scale, mapped_center[1] / scale)
            results.append({"score": float(max_val), "angle": float(angle), "position": mapped_center})
            if candidates_per_angle > 1:
                cv2.floodFill(result, None, max_loc, -1)
            else:
                break
    results.sort(key=lambda x: -x["score"])
    return results

def match_template_rotated_coarse_to_fine(src, template, coarse_step=5, fine_step=1, coarse_range=(-15, 15), fine_window=2, candidates_per_angle=1, min_score=0.5, scale=None, iou_thresh=0.3):
    if scale is None:
        scale = guess_best_scale_threaded_cached(
            src, template,
            scales=[0.01, 0.05, 0.1, 0.15, 0.25, 0.3],
            min_score=min_score,
            num_threads=4  # adjust depending on your CPU
        )
        if scale is None:
            return []
    coarse_matches = match_template_rotated_fast(src, template, angle_range=coarse_range, angle_step=coarse_step, candidates_per_angle=candidates_per_angle, min_score=min_score, scale=scale)
    if not coarse_matches:
        return []
    best_angle = coarse_matches[0]['angle']
    fine_range = (best_angle - fine_window, best_angle + fine_window)
    fine_matches = match_template_rotated_fast(src, template, angle_range=fine_range, angle_step=fine_step, candidates_per_angle=candidates_per_angle, min_score=min_score, scale=scale)
    all_matches = coarse_matches + fine_matches
    all_matches.sort(key=lambda x: -x['score'])
    th, tw = template.shape
    return non_maximum_suppression_iou(all_matches, iou_thresh=iou_thresh, template_size=(tw, th))

_template_cache = {}

def _resize_pair(src, template, scale):
    src_scaled = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    key = (id(template), scale)
    if key in _template_cache:
        template_scaled = _template_cache[key]
    else:
        template_scaled = cv2.resize(template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _template_cache[key] = template_scaled
    return src_scaled, template_scaled

def _score_scale_pair(args):
    src, template, scale = args
    try:
        src_s, template_s = _resize_pair(src, template, scale)
        if template_s.shape[0] > src_s.shape[0] or template_s.shape[1] > src_s.shape[1]:
            return (scale, -1)
        result = cv2.matchTemplate(src_s, template_s, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return (scale, max_val)
    except:
        return (scale, -1)

def guess_best_scale_threaded_cached(src, template, scales, min_score=0.5, num_threads=4):
    tasks = [(src, template, s) for s in scales]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(_score_scale_pair, tasks))

    best_scale = None
    best_score = -1
    for scale, score in results:
        if score > best_score:
            best_score = score
            best_scale = scale

    # print(f"[Auto finding] Best scale: {best_scale:.4f}, score: {best_score:.3f}")
    return best_scale if best_score >= min_score else None


if __name__ == "__main__":
    src_color = cv2.imread(r"/home/snc/Documents/Ubend/machine_vision/image_demo/image_10.jpg", cv2.IMREAD_COLOR)
    src = cv2.imread(r"/home/snc/Documents/Ubend/machine_vision/image_demo/image_10.jpg", cv2.IMREAD_GRAYSCALE)

    template = cv2.imread(r"/home/snc/Documents/Ubend/machine_vision/image_demo/Template.png", cv2.IMREAD_GRAYSCALE)

    if src is None or template is None:
        raise FileNotFoundError("Missing input images!")

    import time
    start = time.perf_counter()
    matches = match_template_rotated_coarse_to_fine(
        src, template,
        coarse_step=5,  # coarse step finding
        fine_step=1,
        coarse_range=(-360, 360), # angle search range
        fine_window=2,  # step search in best angle
        candidates_per_angle=1, # increase number for detecting multiple object at same angle
        min_score = 0.5,
        scale = 0.1,  # Scale is big, cost time, more accuracy & vise-versa
        iou_thresh = 0.2 # overlay ratio
    )
    end = time.perf_counter()

    vis = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    th, tw = template.shape
    for i, match in enumerate(matches):
        pos = match["position"]
        angle = match["angle"]
        score = match["score"]
        # print(pos, angle)
        draw_rotated_match(vis, pos, tw, th, angle, label=f"#{i} {angle:.1f}° {score:.2f}")
        rotated_src, rot_mat = rotate_image(src_color, angle, border_value=255)
        center_in_rot = apply_affine_transform(pos, rot_mat)
        cx_r, cy_r = int(round(center_in_rot[0])), int(round(center_in_rot[1]))
        x0 = cx_r - tw // 2
        y0 = cy_r - th // 2
        x1 = x0 + tw
        y1 = y0 + th
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - rotated_src.shape[1])
        pad_bottom = max(0, y1 - rotated_src.shape[0])

        crop_x0 = max(0, x0)
        crop_y0 = max(0, y0)
        crop_x1 = min(rotated_src.shape[1], x1)
        crop_y1 = min(rotated_src.shape[0], y1)
        if crop_x1 > crop_x0 and crop_y1 > crop_y0:
            crop = rotated_src[crop_y0:crop_y1, crop_x0:crop_x1]
            if pad_top or pad_bottom or pad_left or pad_right:
                crop = cv2.copyMakeBorder(
                    crop,
                    top=pad_top, bottom=pad_bottom,
                    left=pad_left, right=pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=255
                )

    # print(f"Found {len(matches)} matches in {end - start:.3f}s")
    cv2.namedWindow("Rotated Matches", cv2.WINDOW_NORMAL)
    cv2.imshow("Rotated Matches", vis)
    rotated_src_resized = cv2.resize(rotated_src, (640, 480))
    cv2.imshow("Rotated Source", rotated_src_resized)
    crop_resized = cv2.resize(crop, (640, 480))
    cv2.imshow("Cropped", crop_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()