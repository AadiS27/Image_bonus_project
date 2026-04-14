import cv2
import numpy as np
from scipy.ndimage import uniform_filter

def apply_gaussian(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_median(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def hybrid_filter_fixed(image, threshold, kernel_size=5):
    gaussian_filtered = apply_gaussian(image, (kernel_size, kernel_size))
    median_filtered = apply_median(image, kernel_size)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.float32)
    
    mean_sq = uniform_filter(gray**2, size=kernel_size)
    sq_mean = uniform_filter(gray, size=kernel_size)**2
    local_variance = np.maximum(mean_sq - sq_mean, 0)
    
    mask = local_variance > threshold
    if len(image.shape) == 3:
        mask = np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2)
        
    final_image = np.where(mask, median_filtered, gaussian_filtered)
    return final_image.astype(np.uint8)

def hybrid_filter_dynamic(image, k_param=1.0, kernel_size=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.float32)
    
    global_variance = np.var(gray)
    threshold = k_param * global_variance
    
    return hybrid_filter_fixed(image, threshold, kernel_size)

def apply_dct(image, k_thresh=0.1):
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)
        return cv2.merge((apply_dct(b, k_thresh), apply_dct(g, k_thresh), apply_dct(r, k_thresh)))

    gray = image
    h, w = gray.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    
    img_f = gray[:new_h, :new_w].astype(np.float32)
    denoised = np.zeros_like(img_f)
    
    for i in range(0, new_h, 8):
        for j in range(0, new_w, 8):
            block = img_f[i:i+8, j:j+8]
            dct_block = cv2.dct(block)
            thresh = k_thresh * np.max(np.abs(dct_block))
            dct_thresh = np.where(np.abs(dct_block) < thresh, 0, dct_block)
            denoised[i:i+8, j:j+8] = cv2.idct(dct_thresh)
            
    final_img = np.zeros_like(gray, dtype=np.uint8)
    final_img[:new_h, :new_w] = np.clip(denoised, 0, 255).astype(np.uint8)
    if new_h < h:
        final_img[new_h:, :] = gray[new_h:, :]
    if new_w < w:
        final_img[:, new_w:] = gray[:, new_w:]
        
    return final_img
