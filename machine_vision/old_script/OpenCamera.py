import sys, os
from ctypes import *
import cv2
import numpy as np

sys.path.append("E:\\99IS\\B6\\ubend\\machine_vision\\MvImport")
from MvCameraControl_class import *

def main():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE

    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("No device found or Enum failed!")
        return

    cam = MvCamera()
    deviceInfo = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(deviceInfo)
    if ret != 0:
        print("Create Handle failed:", ret)
        return

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("Open device failed:", ret)
        return

    # ✅ ตั้งค่ากล้องให้รอ Trigger จากภายนอก
    cam.MV_CC_SetEnumValue("TriggerMode", 1)           # 1 = On
    cam.MV_CC_SetEnumValue("TriggerSource", 0)         # 0 = Line0
    cam.MV_CC_SetEnumValue("TriggerActivation", 2)     # 0 = RisingEdge
    cam.MV_CC_SetEnumValue("PixelFormat", 0x02180014)  # PixelType_RGB8_Packed
    cam.MV_CC_SetFloatValue("TriggerDelay", 20000)  # 5000 us

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing failed:", ret)
        return

    print("Waiting for trigger signal from sensor...")

    cdll.LoadLibrary("MvCameraControl.dll")
    memcpy = cdll.msvcrt.memcpy
    counter = 0
    while True:
        stData = MV_FRAME_OUT()
        ret = cam.MV_CC_GetImageBuffer(stData, 2000)
        if ret == 0:
            print(f"Got image from trigger: {stData.stFrameInfo.nWidth}x{stData.stFrameInfo.nHeight}")
            counter += 1
            frame_data = (c_ubyte * stData.stFrameInfo.nFrameLen)()
            memcpy(byref(frame_data), stData.pBufAddr, stData.stFrameInfo.nFrameLen)

            img = np.frombuffer(frame_data, dtype=np.uint8)
            img = img.reshape((stData.stFrameInfo.nHeight, stData.stFrameInfo.nWidth, 3))
            img = cv2.resize(img, (640, 480))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            text = f"Count: {counter}"
            cv2.putText(img_bgr, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("Hikrobot Triggered", img_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cam.MV_CC_FreeImageBuffer(stData)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        else:
            print("Waiting for trigger...")

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
