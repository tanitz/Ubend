import sys
import os
import struct
import cv2
from MvImport.CameraParams_header import MV_CC_DEVICE_INFO_LIST
import numpy as np
from PySide6.QtCore import QThread, Signal

# --- ส่วนสำหรับกล้อง Hikrobot ---
HIK_AVAILABLE = False
base_dir = os.path.dirname(os.path.abspath(__file__))

# เลือก 64/32 ตาม Python process
bits = struct.calcsize('P') * 8
arch = "64" if bits == 64 else "32"

# paths ที่จะค้นหา (ให้โฟลเดอร์ project/MvImport เป็นหลัก แล้ว fallback ไป /opt ที่สอดคล้องกับ arch)
candidates = [
    os.path.join(base_dir, "..", "MvImport"),
    f"/opt/MVS/Samples/{arch}/Python/MvImport",
]

# เพิ่ม LD_LIBRARY_PATH ให้ชี้ไปยังไลบรารีของสถาปัตยกรรมที่ถูกต้อง (prepend)
lib_dir = f"/opt/MVS/lib/{arch}"
if os.path.isdir(lib_dir):
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in prev.split(":"):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + prev if prev else "")
        try:
            os.putenv("LD_LIBRARY_PATH", os.environ["LD_LIBRARY_PATH"])
        except Exception:
            pass

# เพิ่มพาธที่มีอยู่ไปยัง sys.path (นำหน้า)
for p in candidates:
    p = os.path.abspath(p)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from MvCameraControl_class import *
    HIK_AVAILABLE = True
except Exception as e:
    print("Warning: cannot import MvCameraControl_class:", e)
    print("Searched MvImport in:", [os.path.abspath(p) for p in candidates])
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    HIK_AVAILABLE = False

class USBCameraWorker(QThread):
    """Worker สำหรับกล้อง USB ทั่วไป"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, device_index=0, backend=cv2.CAP_V4L2):
        super().__init__()
        self.running = True
        self.cap = None
        self.device_index = device_index
        self.backend = backend

    def run(self):
        # ลองเปิดหลายดัชนี ถ้า index เริ่มต้นไม่สำเร็จ
        tried = []
        max_try = 4
        for idx in range(self.device_index, self.device_index + max_try):
            tried.append(idx)
            self.cap = cv2.VideoCapture(idx + 0, self.backend)
            if self.cap.isOpened():
                print(f"USB camera opened on index {idx} (backend={self.backend})")
                break
            else:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        if self.cap is None or not self.cap.isOpened():
            msg = f"ไม่สามารถเปิดกล้อง USB (tried indices: {tried}). ตรวจสอบ /dev/video* และสิทธิ์."
            print(msg)
            self.error_occurred.emit(msg)
            return

        # เพิ่ม debug ขนาดเฟรมจริง
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"USB camera properties: {w}x{h} fps={fps}")

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                # ถ้าอ่านเฟรมล้มเหลว ให้ log แต่ไม่หยุดทันที
                print("Warning: USB camera read() returned False")
            self.msleep(30)

        try:
            self.cap.release()
        except Exception:
            pass
        print("USB Camera worker stopped.")

    def stop(self):
        self.running = False

class HikrobotCameraWorker(QThread):
    """Worker สำหรับกล้อง Hikrobot (ใช้ตัวอย่าง SDK เป็นต้นแบบ)"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, device_index=0, timeout_ms=2000):
        super().__init__()
        self.running = True
        self.cam = None
        self.device_index = device_index
        self.timeout_ms = int(timeout_ms)

    def run(self):
        # init SDK
        try:
            MvCamera.MV_CC_Initialize()
        except Exception as e:
            self.error_occurred.emit(f"Hikrobot: SDK initialize error: {e}")
            return

        try:
            # enumerate devices (support GigE & USB)
            deviceList = MV_CC_DEVICE_INFO_LIST()
            # find ip
            
            
            tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE)
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0:
                raise RuntimeError(f"EnumDevices failed ret={ret}")
            if deviceList.nDeviceNum == 0:
                raise RuntimeError("no Hikrobot device found")

            # pick device_index (clamp)
            idx = max(0, min(self.device_index, deviceList.nDeviceNum - 1))
            deviceInfo = cast(deviceList.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents

            # create camera instance & handle
            self.cam = MvCamera()
            ret = self.cam.MV_CC_CreateHandle(deviceInfo)
            if ret != 0:
                raise RuntimeError(f"CreateHandle failed {ret}")

            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                raise RuntimeError(f"OpenDevice failed {ret}")

            # if GigE, try set optimal packet size
            if deviceInfo.nTLayerType == MV_GIGE_DEVICE or deviceInfo.nTLayerType == MV_GENTL_GIGE_DEVICE:
                try:
                    nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                    if int(nPacketSize) > 0:
                        self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(nPacketSize))
                except Exception:
                    pass

            # set trigger off (continuous)
            try:
                # use SDK constants if available; fallback numeric if not
                self.cam.MV_CC_SetEnumValue("TriggerMode", 1)           # 1 = On
                self.cam.MV_CC_SetEnumValue("TriggerSource", 0)         # 0 = Line0
                self.cam.MV_CC_SetEnumValue("TriggerActivation", 0)     # 0 = RisingEdge
                self.cam.MV_CC_SetEnumValue("PixelFormat", 0x02180014)  # PixelType_RGB8_Packed
                #Exposure time
                self.cam.MV_CC_SetIntValue("ExposureTime", 5000)  # Set exposure time to 500us
                #trigger delay set
                self.cam.MV_CC_SetIntValue("TriggerDelay", 1)  # Set trigger delay to 1us
            except Exception:
                try:
                    self.cam.MV_CC_SetEnumValue("TriggerMode", 0)
                except Exception:
                    pass

            # get payload size
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(stParam))
            ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0 or stParam.nCurValue <= 0:
                raise RuntimeError(f"Get PayloadSize failed {ret}")
            nPayloadSize = int(stParam.nCurValue)

            # start grabbing
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                raise RuntimeError(f"StartGrabbing failed {ret}")

            # capture loop
            stData = MV_FRAME_OUT()
            while self.running and not self.isInterruptionRequested():
                ret = self.cam.MV_CC_GetImageBuffer(stData, self.timeout_ms)
                if ret == 0:
                    try:
                        fh = int(stData.stFrameInfo.nHeight)
                        fw = int(stData.stFrameInfo.nWidth)
                        fl = int(stData.stFrameInfo.nFrameLen)
                        # create buffer and copy
                        frame_data = (c_ubyte * fl)()
                        memmove(byref(frame_data), stData.pBufAddr, fl)
                        arr = np.frombuffer(frame_data, dtype=np.uint8)
                        # try to reshape to H,W,3; if pixel format different, caller may adapt
                        if arr.size >= fh * fw * 3:
                            img = arr[:fh * fw * 3].reshape((fh, fw, 3))
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        else:
                            # fallback to single-channel or whatever available
                            img_bgr = arr.reshape((fh, fw))
                        self.frame_ready.emit(img_bgr)
                        QThread.msleep(10)
                    except Exception as e:
                        print("Hikrobot frame processing error:", e)
                    finally:
                        try:
                            self.cam.MV_CC_FreeImageBuffer(stData)
                        except Exception:
                            pass
                else:
                    # non-zero ret = timeout or error
                    # print debug but continue
                    # print("MV_CC_GetImageBuffer ret=", ret)
                    # short sleep to avoid tight loop when no data
                    self.msleep(5)
        except Exception as e:
            msg = f"Hikrobot init/capture error: {e}"
            print(msg)
            self.error_occurred.emit(msg)
        finally:
            # cleanup always
            try:
                if self.cam:
                    self.cam.MV_CC_StopGrabbing()
                    self.cam.MV_CC_CloseDevice()
                    self.cam.MV_CC_DestroyHandle()
            except Exception as e:
                print("Hikrobot cleanup error:", e)
            print("Hikrobot worker stopped and cleaned up.")

    def stop(self):
        self._running = False
        self.requestInterruption()
        self.quit()
