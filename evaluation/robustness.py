"""
Robustness Perturbation Evaluator (Illumination, Blur, Background Noise, Geometric)
"""
import cv2
import numpy as np

def apply_motion_blur(img, severity=15):
    kernel = np.zeros((severity, severity))
    kernel[int((severity - 1) / 2), :] = np.ones(severity)
    kernel /= severity
    return cv2.filter2D(img, -1, kernel)

def apply_illumination(img, mode='low'):
    if mode == 'low':
        return cv2.convertScaleAbs(img, alpha=0.4, beta=-30)
    elif mode == 'bright':
        return cv2.convertScaleAbs(img, alpha=1.5, beta=40)
    elif mode == 'warm':
        res = img.copy()
        res[:, :, 2] = np.clip(res[:, :, 2].astype(int) + 40, 0, 255)
        return res
    elif mode == 'cool':
        res = img.copy()
        res[:, :, 0] = np.clip(res[:, :, 0].astype(int) + 40, 0, 255)
        return res
    elif mode == 'backlit':
        return cv2.convertScaleAbs(img, alpha=0.5, beta=-15)
    return img

def apply_geometric_perturbation(img, angle_deg=10, scale=0.9, tx=10, ty=10):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
