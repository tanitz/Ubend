import configparser
import os

class ConfigManager:
    """
    คลาสสำหรับจัดการการตั้งค่าโดยใช้ไฟล์ config.ini
    """
    def __init__(self):
        self.config_file = "config.ini"
        self.config = configparser.ConfigParser()
        
        # --- ค่าเริ่มต้น ---
        self.camera_source = "USB"
        
        # --- โหลดค่าที่เคยบันทึกไว้ ---
        self.load_config()

    def load_config(self):
        """
        โหลดค่าการตั้งค่าจากไฟล์ config.ini
        """
        if not os.path.exists(self.config_file):
            print(f"'{self.config_file}' not found. Creating with default settings.")
            self.save_config() # ถ้าไม่พบไฟล์ ให้สร้างใหม่
            return

        try:
            self.config.read(self.config_file)
            
            # อ่านค่าจาก Section 'Camera' และ Key 'source'
            source = self.config.get('Camera', 'source', fallback="USB")
            if source in ["USB", "Hikrobot"]:
                self.camera_source = source
            else:
                self.camera_source = "USB" # ถ้าค่าไม่ถูกต้อง ให้ใช้ค่าเริ่มต้น
            
            print(f"Loaded config: camera_source = {self.camera_source}")

        except Exception as e:
            print(f"Error loading config file: {e}. Using default settings.")
            self.save_config()

    def save_config(self):
        """
        บันทึกค่า camera_source ปัจจุบันลงในไฟล์ config.ini
        """
        try:
            # สร้าง Section 'Camera' ถ้ายังไม่มี
            if not self.config.has_section('Camera'):
                self.config.add_section('Camera')
            
            # ตั้งค่า
            self.config.set('Camera', 'source', self.camera_source)
            
            # เขียนลงไฟล์
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            
            print(f"Saved config: camera_source = {self.camera_source}")
        except Exception as e:
            print(f"Error saving config file: {e}")

