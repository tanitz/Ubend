import sys, os
sys.path.append("E:\\99IS\\Hikrobot\\MvImport")
from MvCameraControl_class import *
import cv2
import numpy as np
# -- coding: utf-8 --
import sys
import sys, os
from tkinter import ttk
from MvCameraControl_class import *
# from CamOperation_class import *
from PIL import Image,ImageTk
import cv2

def Color_numpy(data,nWidth,nHeight):
    data_ = np.frombuffer(data, count=int(nWidth*nHeight*3), dtype=np.uint8, offset=0)
    data_r = data_[0:nWidth*nHeight*3:3]
    data_g = data_[1:nWidth*nHeight*3:3]
    data_b = data_[2:nWidth*nHeight*3:3]
    data_r_arr = data_r.reshape(nHeight, nWidth)
    data_g_arr = data_g.reshape(nHeight, nWidth)
    data_b_arr = data_b.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 3],"uint8")
    numArray[:, :, 0] = data_r_arr
    numArray[:, :, 1] = data_g_arr
    numArray[:, :, 2] = data_b_arr
    return numArray

# torch.cuda.set_device(0)

# model = attempt_load('yolov8s.pt', map_location=device)
global deviceList 
deviceList = MV_CC_DEVICE_INFO_LIST()
global tlayerType
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
global obj_cam_operation
obj_cam_operation = 0
global nOpenDevSuccess
nOpenDevSuccess = 0
global devList
devList = []
nOpenDevSuccess = 0
obj_cam_operation = []
deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
mvcc_dev_info = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
devList.append("Gige["+str(0)+"]:"+str(nip1)+"."+str(nip2)+"."+str(nip3)+"."+str(nip4))
cam = MvCamera()
stDeviceList = cast(deviceList.pDeviceInfo[int(0)], POINTER(MV_CC_DEVICE_INFO)).contents
ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print ("create handle fail! ret[0x%x]" % ret)
    sys.exit()
ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != 0:
    print ("open device fail! ret[0x%x]" % ret)
if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
    nPacketSize = cam.MV_CC_GetOptimalPacketSize()
    if int(nPacketSize) > 0:
        ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
        if ret != 0:
            print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
    else:
        print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
stBool = c_bool(False)
ret =cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
if ret != 0:
    print ("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
if ret != 0:
    print ("set trigger mode fail! ret[0x%x]" % ret)
ret = cam.MV_CC_StartGrabbing()
if ret != 0:
    print ("start grabbing fail! ret[0x%x]" % ret)
print(ret)
obj_cam_operation.append(CameraOperation(cam,deviceList,0))
ret = obj_cam_operation[nOpenDevSuccess].Open_device()
cam.MV_CC_SetFloatValue("ExposureTime",float(150000))
stOutFrame = MV_FRAME_OUT() 
memset(byref(stOutFrame), 0, sizeof(stOutFrame))
img_buff = None
buf_cache = None    

prev_frame_time = 0
new_frame_time = 0

while True:

    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 5000)
    if 0 == ret:
        if buf_cache is None:
            buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
        st_frame_info = stOutFrame.stFrameInfo
        cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, st_frame_info.nFrameLen)
        # print ("Camera[%d]:get one frame: Width[%d], Height[%d], nFrameNum[%d]"  % (0,st_frame_info.nWidth, st_frame_info.nHeight, st_frame_info.nFrameNum))
        n_save_image_size = st_frame_info.nWidth * st_frame_info.nHeight * 3 + 2048
        if img_buff is None:
            img_buff = (c_ubyte * n_save_image_size)()
    stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
    memset(byref(stConvertParam), 0, sizeof(stConvertParam))
    stConvertParam.nWidth = st_frame_info.nWidth
    stConvertParam.nHeight = st_frame_info.nHeight
    stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
    stConvertParam.nSrcDataLen = st_frame_info.nFrameLen
    stConvertParam.enSrcPixelType = st_frame_info.enPixelType 
    if PixelType_Gvsp_RGB8_Packed == st_frame_info.enPixelType:
        numArray = CameraOperation.Color_numpy(buf_cache,st_frame_info.nWidth,st_frame_info.nHeight)
    else:
        nConvertSize = st_frame_info.nWidth * st_frame_info.nHeight * 3
        stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
        stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
        stConvertParam.nDstBufferSize = nConvertSize
        ret = cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
        if ret != 0:
            continue
        cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
        numArray = Color_numpy(img_buff,st_frame_info.nWidth,st_frame_info.nHeight)
        image= cv2.cvtColor(numArray,cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image,(1920,1080))
        # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_frame_time = time.time()
        # frame = cv2.resize(image,(1920,1080))
        
        # frame = torch.from_numpy(frame).to(device)
        show = cv2.resize(image,(640,480))
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("image",cv2.rotate(show, cv2.ROTATE_90_COUNTERCLOCKWISE))
        if cv2.waitKey(1) == 32:
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            image_filename = f'img\{timestamp}.jpg'
            cv2.imwrite(image_filename ,image)
            print(f"image captureed and saved as '{image_filename}'")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # cv2.waitKey(1)
    nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
cv2.destroyAllWindows()