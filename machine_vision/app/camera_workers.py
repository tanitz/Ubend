import sys
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

# --- ส่วนสำหรับกล้อง Hikrobot ---
try:
    # ❗️❗️ โปรดแก้ไข Path นี้ให้ตรงกับตำแหน่งของ SDK ในเครื่องของคุณ ❗️❗️
    sys.path.append("E:\\99IS\\B6\\ubend\\machine_vision\\MvImport")
    from MvCameraControl_class import *
    HIK_AVAILABLE = True
except ImportError:
    print("Warning: ไม่พบไลบรารี MvImport, โหมด Hikrobot จะไม่สามารถใช้งานได้")
    HIK_AVAILABLE = False

class USBCameraWorker(QThread):
    """Worker สำหรับกล้อง USB ทั่วไป"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error_occurred.emit("ไม่สามารถเปิดกล้อง USB ได้")
            return
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(30) # หน่วงเวลาเล็กน้อยเพื่อลดการใช้ CPU

        self.cap.release()
        print("USB Camera worker stopped.")

    def stop(self):
        self.running = False

class HikrobotCameraWorker(QThread):
    """Worker สำหรับกล้อง Hikrobot"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.cam = None

    def run(self):
        if not HIK_AVAILABLE:
            self.error_occurred.emit("ไม่พบไลบรารี MvImport ของ Hikrobot")
            return

        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            self.error_occurred.emit("ไม่พบอุปกรณ์กล้อง Hikrobot")
            return

        self.cam = MvCamera()
        deviceInfo = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(deviceInfo)
        if ret != 0:
            self.error_occurred.emit(f"Hikrobot: Create Handle ล้มเหลว: {ret}")
            return

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.error_occurred.emit(f"Hikrobot: Open device ล้มเหลว: {ret}")
            return

        self.cam.MV_CC_SetEnumValue("TriggerMode", 1)
        self.cam.MV_CC_SetEnumValue("TriggerSource", 0)
        self.cam.MV_CC_SetEnumValue("PixelFormat", 0x02180014)
        
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.error_occurred.emit(f"Hikrobot: Start grabbing ล้มเหลว: {ret}")
            self.cleanup()
            return

        stData = MV_FRAME_OUT()
        while self.running:
            ret = self.cam.MV_CC_GetImageBuffer(stData, 2000)
            if ret == 0:
                frame_data = (c_ubyte * stData.stFrameInfo.nFrameLen)()
                memmove(byref(frame_data), stData.pBufAddr, stData.stFrameInfo.nFrameLen)
                img = np.frombuffer(frame_data, dtype=np.uint8).reshape((stData.stFrameInfo.nHeight, stData.stFrameInfo.nWidth, 3))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.frame_ready.emit(img_bgr)
                self.cam.MV_CC_FreeImageBuffer(stData)

        self.cleanup()

    def cleanup(self):
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
        print("Hikrobot worker stopped and cleaned up.")

    def stop(self):
        self.running = False
