import cv2
import numpy as np

# --- 1. โหลดและเตรียมรูปภาพ ---
try:
    # !!! กรุณาใส่ชื่อไฟล์รูปภาพของคุณที่นี่ !!!
    file_name = '1.png' 
    img = cv2.imread(file_name)
    
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพ '{file_name}'")
    
    # สร้างสำเนาของรูปภาพเพื่อวาดผลลัพธ์
    img_result = img.copy()

    # แปลงภาพเป็น Grayscale และทำ Binary Threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img_gray, 127, 255, 0)

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")
    exit()

# --- 2. ค้นหาเส้นขอบ (Contour) ---
contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # เลือกเส้นขอบที่ใหญ่ที่สุด
    cnt = max(contours, key=cv2.contourArea)

    # --- 3. ใช้วิธี approxPolyDP เพื่อหามุม ---
    
    # พารามิเตอร์ epsilon คือค่าความคลาดเคลื่อนที่ยอมรับได้
    # ยิ่งค่านี้น้อย จะยิ่งตรวจจับมุมได้เยอะขึ้น (ไวขึ้น)
    # โดยทั่วไปจะใช้เป็น % ของความยาวเส้นขอบทั้งหมด (arcLength)
    # ลองปรับค่า 0.01, 0.02, 0.005 เพื่อหาผลลัพธ์ที่ดีที่สุด
    epsilon = 0.005 * cv2.arcLength(cnt, False)

    
    # ฟังก์ชัน approxPolyDP จะคืนค่าเป็นชุดของจุดที่เป็นมุม
    corners = cv2.approxPolyDP(cnt, epsilon, True)
    
    print(f"ตรวจพบมุมทั้งหมด {len(corners)} จุด")

    # วาดวงกลมที่มุมที่ตรวจพบ
    for corner in corners:
        # corner[0][0] คือ x, corner[0][1] คือ y
        point = tuple(corner[0])
        cv2.circle(img_result, point,2, (0, 0, 255), -1) # วาดวงกลมสีแดง

    # --- 4. บันทึกและแสดงผลลัพธ์ --
    cv2.imshow('Detected Corners', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("ไม่พบเส้นขอบในรูปภาพ")