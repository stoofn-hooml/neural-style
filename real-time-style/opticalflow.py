import cv2
import numpy as np

"""
This module is unused in our final implementation
Modified from curaai00 on Github: https://github.com/curaai00/RT-StyleTransfer-forVideo/blob/master/opticalflow.py
"""

def opticalflow(generated_t, generated_t1):
    generated_t_gray = cv2.cvtColor(generated_t, cv2.COLOR_RGB2GRAY)
    generated_t1_gray = cv2.cvtColor(generated_t1, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(generated_t_gray, generated_t1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv = np.zeros_like(generated_t)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.where(gray > 0.5, 0, 1)

    return rgb, mask
