# ภาพจาก IMAGE 3 จะถูกส่งเข้ามาในตัวแปร 'input_image'
# ผลลัพธ์สุดท้ายต้องถูกเก็บไว้ในตัวแปร 'output_image'

hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)
s_increased = cv2.add(s, 10)
s_decreased = cv2.subtract(s, 10)

_, object_mask = cv2.threshold(s_increased, 50, 255, cv2.THRESH_BINARY)
_, hole_mask = cv2.threshold(s_increased, 150, 255, cv2.THRESH_BINARY_INV)
inpaint_mask = cv2.bitwise_and(object_mask, hole_mask)
result = cv2.inpaint(s_increased, inpaint_mask, 5, cv2.INPAINT_TELEA)
contours, hierarchy = cv2.findContours(inpaint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_image = np.zeros_like(inpaint_mask)
cv2.drawContours(filled_image, contours, -1, (255), cv2.FILLED)
blurred = cv2.GaussianBlur(filled_image, (15, 15), 0)
_, smoothed_img = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(smoothed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
smoothed_img = cv2.cvtColor(smoothed_img, cv2.COLOR_GRAY2BGR)
#cv2.drawContours(smoothed_img, contours, -1, (0, 255, 0), 2)

main_contour = max(contours, key=cv2.contourArea)
x_b, y_b, w_b, h_b = cv2.boundingRect(main_contour)
all_points = main_contour[:, 0, :]

widths_at_y = []
flange_y_coords = []

width_change_threshold = 1.0

for i in range(1, len(widths_at_y)):
 prev_width = widths_at_y[i-1]['width']
 curr_width = widths_at_y[i]['width']
 if (curr_width > prev_width * width_change_threshold) or (prev_width > curr_width * width_change_threshold):
  flange_y_coords.append(widths_at_y[i]['y'])

if len(flange_y_coords) >= 0:
 y_min = min(flange_y_coords)
 y_max = max(flange_y_coords)
 points_in_flange_y_range = all_points[(all_points[:, 1] >= y_min) & (all_points[:, 1] <= y_max)]
 x_min = np.min(points_in_flange_y_range[:, 0])
 x_max = np.max(points_in_flange_y_range[:, 0])
 cv2.rectangle(smoothed_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2) 


output_image = smoothed_img