import cv2
import numpy as np
import glob
import os
import time

IMAGE_DIR = '/home/snc/Documents/Ubend/machine_vision/image_rotated'

def find_oring(image: np.ndarray,
               min_area: int = 500,
               delta: int = 20,
               width_change_threshold: float = 1.1) -> np.ndarray:
    """
    หา O-ring แล้ววาดผลลัพธ์ทั้งหมด ส่งคืนภาพสุดท้ายพร้อมแสดงผล
    - รวมงานเดิมของ find_oring + make_display
    - ซ้าย: ภาพต้นฉบับ, ขวา: ภาพประมวลผลพร้อมกล่อง/ตัวเลข/ตัวชี้วัด
    """
    # ตรวจสอบ input
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros((400, 600, 3), dtype=np.uint8)

    # สำเนาภาพต้นฉบับ
    img = image.copy()

    # 1) สร้าง mask เบื้องต้นจากช่อง S (Saturation) และ inpaint
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_increased = cv2.add(s, 10)

    _, object_mask = cv2.threshold(s_increased, 0, 100, cv2.THRESH_BINARY)
    _, hole_mask = cv2.threshold(s_increased, 100, 255, cv2.THRESH_BINARY_INV)
    inpaint_mask = cv2.bitwise_and(object_mask, hole_mask)
    try:
        _ = cv2.inpaint(s_increased, inpaint_mask, 1, cv2.INPAINT_NS)
    except Exception:
        pass

    contours, _ = cv2.findContours(inpaint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(inpaint_mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    # 2) ทำให้ mask เนียนขึ้น
    blurred = cv2.GaussianBlur(filled, (15, 15), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours2, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3) เลือกเฉพาะคอนทัวร์ใหญ่
    large_contours = [c for c in contours2 if cv2.contourArea(c) >= min_area]

    # 4) วาดเส้นขอบชิ้นงานทับบนภาพต้นฉบับ แล้ว blend ให้เห็นชัด
    overlay = img.copy()
    if large_contours:
        cv2.drawContours(overlay, large_contours, -1, (0, 0, 200), 5)
    p = cv2.addWeighted(img, 0.7, overlay, 0.5, 0)

    # 5) วิเคราะห์เฉพาะโซนล่าง (3 และ 4) และวาดกล่อง/ข้อความผลลัพธ์บน p
    hh, ww = p.shape[:2]
    x_mid = ww // 2
    y_mid = int(np.clip(hh // 2 - delta, 0, hh))
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_color = (0, 0, 255)

    pts = None
    if large_contours:
        try:
            pts = np.vstack(large_contours)[:, 0, :].astype(np.int32)
        except Exception:
            pts = None

    zones = {
        3: (0, x_mid, y_mid, hh),   # bottom-left
        4: (x_mid, ww, y_mid, hh),  # bottom-right
    }
    avg_ys = {}
    zone_y_maxs = {}

    for z, (x0, x1, y0, y1) in zones.items():
        if pts is None or pts.size == 0:
            continue

        mask_zone = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        pts_zone = pts[mask_zone]
        if pts_zone.size == 0:
            continue

        x_min = int(np.min(pts_zone[:, 0]))
        x_max = int(np.max(pts_zone[:, 0]))
        y_max = int(np.max(pts_zone[:, 1]))

        y_at_x_min = int(np.median(pts_zone[pts_zone[:, 0] == x_min][:, 1])) if np.any(pts_zone[:, 0] == x_min) else int(np.median(pts_zone[:, 1]))
        y_at_x_max = int(np.median(pts_zone[pts_zone[:, 0] == x_max][:, 1])) if np.any(pts_zone[:, 0] == x_max) else int(np.median(pts_zone[:, 1]))
        x_at_y_max = int(np.median(pts_zone[pts_zone[:, 1] == y_max][:, 0])) if np.any(pts_zone[:, 1] == y_max) else int(np.median(pts_zone[:, 0]))

        avg_y = (y_at_x_min + y_at_x_max) // 2
        avg_ys[z] = avg_y
        zone_y_maxs[z] = y_max

        # จุด reference ด้านล่าง
        cv2.circle(p, (x_at_y_max, y_max), 6, (255, 0, 0), -1)

        rect_width = x_max - x_min
        rect_height = max(1, int(0.5 * rect_width))
        top_y = max(0, avg_y - rect_height // 2)
        bottom_y = min(hh - 1, avg_y + rect_height // 2)

        diff_val = abs(avg_y - y_max)
        fill_color = (0, 255, 0) if (diff_val > 65 and rect_width > 110) else (0, 0, 255)

        overlay2 = p.copy()
        cv2.rectangle(overlay2, (x_min, top_y), (x_max, bottom_y), fill_color, -1)
        p = cv2.addWeighted(overlay2, 0.4, p, 0.6, 0)
        cv2.rectangle(p, (x_min, top_y), (x_max, bottom_y), (255, 0, 0), 1)

        # ความกว้าง
        length_text = f"{rect_width}px"
        text_scale, text_thickness = 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(length_text, font, text_scale, text_thickness)
        text_x = x_min + max(0, (rect_width - text_w) // 2)
        text_y = top_y - 6
        if text_y < text_h + 4:
            text_y = top_y + text_h + 4
        cv2.putText(p, length_text, (text_x, text_y), font, text_scale, label_color, text_thickness, cv2.LINE_AA)

        # diff ต่อโซน
        info_text = f"diff:{diff_val}px"
        info_scale, info_thickness = 0.5, 1
        (iw, ih), _ = cv2.getTextSize(info_text, font, info_scale, info_thickness)
        info_x = x_min + 4
        info_y = bottom_y + ih + 6
        if info_y > hh - 4:
            info_y = top_y - 6
            if info_y < ih + 4:
                info_y = top_y + ih + 4
        cv2.putText(p, info_text, (info_x, info_y), font, info_scale, (255, 0, 0), info_thickness, cv2.LINE_AA)

    if 3 in avg_ys and 4 in avg_ys:
        diff = int(avg_ys[4] - avg_ys[3])
        cv2.putText(p, f"Δavg_y (Z4 - Z3): {diff}px", (10, 25), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 6) รวมภาพต้นฉบับ (ซ้าย) + ภาพผล (ขวา) ในฟังก์ชันนี้เลย (แทน make_display)
    o = img.copy()
    h1, w1 = o.shape[:2]
    h2, w2 = p.shape[:2]
    max_width = 1400
    scale = 1.0
    if (w1 + w2) > max_width:
        scale = max_width / (w1 + w2)
    new_w1, new_h1 = int(w1 * scale), int(h1 * scale)
    new_w2, new_h2 = int(w2 * scale), int(h2 * scale)
    o_resized = cv2.resize(o, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
    p_resized = cv2.resize(p, (new_w2, new_h2), interpolation=cv2.INTER_AREA)

    H = max(o_resized.shape[0], p_resized.shape[0])
    canvas_o = np.zeros((H, o_resized.shape[1], 3), dtype=o_resized.dtype)
    canvas_p = np.zeros((H, p_resized.shape[1], 3), dtype=p_resized.dtype)
    canvas_o[0:o_resized.shape[0], 0:o_resized.shape[1]] = o_resized
    canvas_p[0:p_resized.shape[0], 0:p_resized.shape[1]] = p_resized
    combined = np.hstack([canvas_o, canvas_p])

    return p_resized

def process_image(path):
    start = time.perf_counter()
    img = cv2.imread(path)
    if img is None:
        return None, [], "cannot open", 0.0

    # find_oring ยังรับแค่ภาพ และคืนภาพรวมสุดท้าย (แทน make_display)
    final_img = find_oring(img)

    elapsed = (time.perf_counter() - start) * 1000.0
    return final_img, [], "ok", elapsed

def list_images(directory):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    paths = []
    for e in exts:
        paths.extend(sorted(glob.glob(os.path.join(directory, e))))
    return paths

# ลบ make_display ออก (ถ้ามี) และให้ main ใช้ภาพจาก find_oring โดยตรง
def main():
    imgs = list_images(IMAGE_DIR)
    if not imgs:
        print("No images in", IMAGE_DIR)
        return
    idx = 0
    win = "Demo Oring - n/p or ←/→ : next/prev | s: save processed | q: quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        path = imgs[idx]
        proc, _lc, status, elapsed_ms = process_image(path)
        if proc is None:
            print("Cannot open", path)
            proc = np.zeros((400, 600, 3), dtype=np.uint8)
            status = "error"
            elapsed_ms = 0.0

        cv2.imshow(win, proc)
        key = cv2.waitKey(0)

        if key == ord('q') or key == 27:  # q or ESC
            break
        if key == ord('n') or key == 83 or key == 2555904:  # n or right arrow
            idx = (idx + 1) % len(imgs)
            continue
        if key == ord('p') or key == 81 or key == 2424832:  # p or left arrow
            idx = (idx - 1) % len(imgs)
            continue
        if key == ord('s'):
            save_dir = IMAGE_DIR
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.basename(path)
            out_path = os.path.join(save_dir, f"processed_{base}")
            cv2.imwrite(out_path, proc)
            print("Saved:", out_path)
            continue

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()