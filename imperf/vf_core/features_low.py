import cv2
import numpy as np
from typing import Dict

# ---- hue stats ----

def hue_stats(img_bgr: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[...,0].astype(np.float32) * (360.0/180.0)
    s = (hsv[...,1].astype(np.float32))/255.0
    v = (hsv[...,2].astype(np.float32))/255.0
    # circular mean for hue (simple approximation)
    h_rad = np.deg2rad(h)
    mean_angle = np.arctan2(np.mean(np.sin(h_rad)), np.mean(np.cos(h_rad)))
    hue_mean = (np.rad2deg(mean_angle) + 360.0) % 360.0
    return {
        'hue_mean': float(hue_mean),
        'hue_std': float(np.std(h)),
        'sat_mean': float(np.mean(s)),
        'val_mean': float(np.mean(v)),
    }

# ---- sharpness (variance of Laplacian) ----

def sharpness_laplacian(img_bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(img_bgr, cv2.CV_64F).var())

# ---- edge orientation histogram ----

def edge_orientation_hist(img_bgr: np.ndarray, bins: int = 8) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ang, bins=bins, range=(0,360), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    return {f'edge_bin_{i}': float(hist[i]) for i in range(bins)}

# ---- spectral residual saliency (fast) ----

def saliency_ratio(img_bgr: np.ndarray, thresh: float = 0.7) -> float:
    # convert to gray float
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # FFT
    fft = np.fft.fft2(gray)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    # average filter in frequency domain
    kernel = cv2.boxFilter(log_amp, ddepth=-1, ksize=(3,3))
    spectral_residual = log_amp - kernel
    saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j*phase)))**2
    saliency = cv2.GaussianBlur(saliency, (9,9), 2.5)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    # ratio of salient pixels
    return float((saliency > thresh).mean())
