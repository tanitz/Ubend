import sys
import ctypes
import numpy as np
import cv2
import os
from glob import glob

# Define MatchResult structure
class MatchResult(ctypes.Structure):
    _fields_ = [
        ('leftTopX', ctypes.c_double),
        ('leftTopY', ctypes.c_double),
        ('leftBottomX', ctypes.c_double),
        ('leftBottomY', ctypes.c_double),
        ('rightTopX', ctypes.c_double),
        ('rightTopY', ctypes.c_double),
        ('rightBottomX', ctypes.c_double),
        ('rightBottomY', ctypes.c_double),
        ('centerX', ctypes.c_double),
        ('centerY', ctypes.c_double),
        ('angle', ctypes.c_double),
        ('score', ctypes.c_double)
    ]

class Matcher:
    def __init__(self, lib_path, maxCount=1, scoreThreshold=0.6, iouThreshold=0.4, angle=360, minArea=256):
        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"Shared library not found: {lib_path}")
        base_candidates = [
            os.path.join(os.path.dirname(lib_path), 'libtemplatematching.so'),
            os.path.join(os.path.dirname(os.path.dirname(lib_path)), 'lib', 'libtemplatematching.so'),
        ]
        for _c in base_candidates:
            if os.path.isfile(_c):
                try:
                    ctypes.CDLL(_c, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        self.lib = ctypes.CDLL(lib_path)
        self.lib.matcher.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.lib.matcher.restype = ctypes.c_void_p
        self.lib.setTemplate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.match.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(MatchResult), ctypes.c_int]
        self.maxCount = maxCount
        self.scoreThreshold = scoreThreshold
        self.iouThreshold = iouThreshold
        self.angle = angle
        self.minArea = minArea
        self.matcher = self.lib.matcher(maxCount, scoreThreshold, iouThreshold, angle, minArea)
        self.results = (MatchResult * self.maxCount)()

    def set_template(self, image):
        height, width = image.shape[0], image.shape[1]
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.setTemplate(self.matcher, data, width, height, channels)

    def match(self, image):
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Invalid image shape")
        height, width = image.shape[0], image.shape[1]
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.match(self.matcher, data, width, height, channels, self.results, self.maxCount)

# Utility function to resolve library path
def resolve_lib_path(project_root=None):
    env_override = os.getenv("TEMPLATE_MATCHING_SO")
    candidate_paths = []
    if env_override:
        candidate_paths.append(env_override)
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
    candidate_paths.extend([
        os.path.join(project_root, 'build', 'lib', 'libtemplatematching_ctype.so'),
        os.path.join(project_root, 'build', 'matcher', 'libtemplatematching_ctype.so'),
        os.path.join(project_root, 'build', 'install', 'lib', 'libtemplatematching_ctype.so'),
    ])
    for p in candidate_paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Could not locate 'libtemplatematching_ctype.so'. Tried paths: " + str(candidate_paths))

# Main matching function for cross-project use
# Usage: matching(template_path, image_path, lib_path=None)
def matching(template_path, image_path, lib_path=None):
    if lib_path is None:
        lib_path = resolve_lib_path()
    matcher = Matcher(lib_path)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Template read failed: {template_path}")
    matcher.set_template(template)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image read failed: {image_path}")
    matches_count = matcher.match(image)
    results = []
    for i in range(min(matches_count, matcher.maxCount)):
        result = matcher.results[i]
        if result.score > 0:
            results.append({
                'score': result.score,
                'angle': result.angle,
                'center': (result.centerX, result.centerY),
                'corners': [
                    (result.leftTopX, result.leftTopY),
                    (result.leftBottomX, result.leftBottomY),
                    (result.rightBottomX, result.rightBottomY),
                    (result.rightTopX, result.rightTopY)
                ]
            })
    return results

if __name__ == '__main__':
    template_path = '/home/snc/Documents/DEMO/opencv_matching/model1.png'
    image_path = '/home/snc/Documents/DEMO/opencv_matching/test/Image_20250916134130468.bmp'
    results = matching(template_path, image_path)
    print(results)
