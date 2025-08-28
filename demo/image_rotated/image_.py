import os

def convert_bmp_to_jpg_in_folder(folder_path):
   
    if not os.path.isdir(folder_path):
        print(f"ข้อผิดพลาด: ไม่พบโฟลเดอร์ '{folder_path}'")
        return

    print(f"กำลังค้นหาไฟล์ในโฟลเดอร์: {folder_path}")
    
    image_counter = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            old_file_path = os.path.join(folder_path, filename)
            
            new_filename = f"image_rotated_{image_counter}.jpg"
            new_file_path = os.path.join(folder_path, new_filename)

            try:
                os.rename(old_file_path, new_file_path)
                print(f"เปลี่ยนชื่อ: '{filename}'  ->  '{new_filename}'")
                image_counter += 1
            except OSError as e:
                print(f"ไม่สามารถเปลี่ยนชื่อไฟล์ '{filename}' ได้: {e}")

    if image_counter == 0:
        print("\nไม่พบไฟล์ .bmp ในโฟลเดอร์นี้")
    else:
        print(f"\nดำเนินการเสร็จสิ้น! เปลี่ยนชื่อไฟล์ทั้งหมด {image_counter} ไฟล์")


target_folder = "/home/snc/Documents/Ubend/machine_vision/image_rotated"  
convert_bmp_to_jpg_in_folder(target_folder)
