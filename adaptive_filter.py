import cv2
import numpy as np
from scipy.ndimage import uniform_filter

def create_gaussian_kernel(size=5, sigma=1.0):
    """Generates a 2D Gaussian kernel manually."""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g / g.sum()

from scipy.ndimage import convolve
from numpy.lib.stride_tricks import sliding_window_view

def apply_gaussian(image, kernel_size=(5, 5), sigma=0):
    k_size = kernel_size[0]
    if sigma <= 0: # OpenCV default formula
        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8 
        
    kernel = create_gaussian_kernel(size=k_size, sigma=sigma)
    
    if len(image.shape) == 3:
        res = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            # Apply convolution manually
            res[:,:,i] = convolve(image[:,:,i].astype(np.float32), kernel, mode='reflect')
        return np.clip(res, 0, 255).astype(np.uint8)
    else:
        # Apply convolution manually
        res = convolve(image.astype(np.float32), kernel, mode='reflect')
        return np.clip(res, 0, 255).astype(np.uint8)

def apply_median(image, kernel_size=5):
    """
    Manual Median Filter:
    1. Sliding window
    2. Sort values
    3. Pick median
    """
    pad = kernel_size // 2
    
    if len(image.shape) == 3:
        res = np.zeros_like(image)
        for c in range(3):
            padded = np.pad(image[:,:,c], pad, mode='reflect')
            # Extract sliding window
            windows = sliding_window_view(padded, (kernel_size, kernel_size))
            windows_flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
            # Sort the array natively
            windows_flat.sort(axis=-1)
            # Pick median element
            mid = (kernel_size * kernel_size) // 2
            res[:,:,c] = windows_flat[:, :, mid]
        return res
    else:
        padded = np.pad(image, pad, mode='reflect')
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        windows_flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
        windows_flat.sort(axis=-1)
        mid = (kernel_size * kernel_size) // 2
        return windows_flat[:, :, mid].astype(np.uint8)

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

def create_dct_matrix(n=8):
    """Generates the 1D DCT transform matrix."""
    c = np.zeros((n, n), dtype=np.float32)
    for u in range(n):
        for v in range(n):
            if u == 0:
                c[u, v] = 1.0 / np.sqrt(n)
            else:
                c[u, v] = np.sqrt(2.0 / n) * np.cos((2 * v + 1) * u * np.pi / (2 * n))
    return c

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
    
    # Precompute DCT transformation matrix
    C = create_dct_matrix(8)
    C_T = C.T
    
    for i in range(0, new_h, 8):
        for j in range(0, new_w, 8):
            block = img_f[i:i+8, j:j+8]
            
            # Manual 2D DCT: C * Block * C^T
            dct_block = C @ block @ C_T
            
            # Thresholding
            thresh = k_thresh * np.max(np.abs(dct_block))
            dct_thresh = np.where(np.abs(dct_block) < thresh, 0, dct_block)
            
            # Manual 2D Inverse DCT: C^T * DCT_Block * C
            denoised[i:i+8, j:j+8] = C_T @ dct_thresh @ C
            
    final_img = np.zeros_like(gray, dtype=np.uint8)
    final_img[:new_h, :new_w] = np.clip(denoised, 0, 255).astype(np.uint8)
    if new_h < h:
        final_img[new_h:, :] = gray[new_h:, :]
    if new_w < w:
        final_img[:, new_w:] = gray[:, new_w:]
        
    return final_img
