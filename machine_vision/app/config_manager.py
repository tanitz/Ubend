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
        # find_oring defaults
        self.large_area = 5000
        self.dif_zone = 0
        self.h_oring = 65
        self.w_oring = 110
        # image processing defaults for find_oring
        self.blur_kernel = 15
        self.canny_thresh1 = 50
        self.canny_thresh2 = 90
        self.morph_kernel = 15
        self.rect_height_ratio = 0.5
        
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
            source = self.config.get('Camera', 'source', fallback=self.camera_source)
            if source in ["USB", "Hikrobot"]:
                self.camera_source = source
            else:
                self.camera_source = "USB" # ถ้าค่าไม่ถูกต้อง ให้ใช้ค่าเริ่มต้น
            
            # อ่านค่าจาก Section 'find_oring'
            if self.config.has_section('find_oring'):
                self.large_area = self.config.getint('find_oring', 'large_area', fallback=self.large_area)
                self.dif_zone = self.config.getint('find_oring', 'dif_zone', fallback=self.dif_zone)
                self.h_oring = self.config.getint('find_oring', 'h_oring', fallback=self.h_oring)
                self.w_oring = self.config.getint('find_oring', 'w_oring', fallback=self.w_oring)
                # optional image processing params (fallback to defaults)
                self.blur_kernel = self.config.getint('find_oring', 'blur_kernel', fallback=self.blur_kernel)
                self.canny_thresh1 = self.config.getint('find_oring', 'canny_thresh1', fallback=self.canny_thresh1)
                self.canny_thresh2 = self.config.getint('find_oring', 'canny_thresh2', fallback=self.canny_thresh2)
                self.morph_kernel = self.config.getint('find_oring', 'morph_kernel', fallback=self.morph_kernel)
                try:
                    self.rect_height_ratio = self.config.getfloat('find_oring', 'rect_height_ratio', fallback=self.rect_height_ratio)
                except Exception:
                    # older config may store as string/int
                    try:
                        self.rect_height_ratio = float(self.config.get('find_oring', 'rect_height_ratio', fallback=str(self.rect_height_ratio)))
                    except Exception:
                        pass
            
            print(f"Loaded config: camera_source = {self.camera_source}, large_area={self.large_area}")

        except Exception as e:
            print(f"Error loading config file: {e}. Using default settings.")
            self.save_config()

    def save_config(self):
        """
        บันทึกค่า camera_source และ find_oring ปัจจุบันลงในไฟล์ config.ini
        """
        try:
            # สร้าง Section 'Camera' ถ้ายังไม่มี
            if not self.config.has_section('Camera'):
                self.config.add_section('Camera')
            if not self.config.has_section('find_oring'):
                self.config.add_section('find_oring')
            
            # ตั้งค่า
            self.config.set('Camera', 'source', self.camera_source)
            self.config.set('find_oring', 'large_area', str(self.large_area))
            self.config.set('find_oring', 'dif_zone', str(self.dif_zone))
            self.config.set('find_oring', 'h_oring', str(self.h_oring))
            self.config.set('find_oring', 'w_oring', str(self.w_oring))
            # write optional image processing params
            self.config.set('find_oring', 'blur_kernel', str(self.blur_kernel))
            self.config.set('find_oring', 'canny_thresh1', str(self.canny_thresh1))
            self.config.set('find_oring', 'canny_thresh2', str(self.canny_thresh2))
            self.config.set('find_oring', 'morph_kernel', str(self.morph_kernel))
            self.config.set('find_oring', 'rect_height_ratio', str(self.rect_height_ratio))
            
            # เขียนลงไฟล์
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            
            print(f"Saved config: camera_source = {self.camera_source}, large_area={self.large_area}")
        except Exception as e:
            print(f"Error saving config file: {e}")

# โหลด ConfigManager หนึ่งครั้งเมื่อ module ถูก import
CONFIG_MANAGER = ConfigManager()

__all__ = ["ConfigManager", "CONFIG_MANAGER"]

