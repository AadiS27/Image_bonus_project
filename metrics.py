import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import cv2

def calculate_mse(original, denoised):
    return mean_squared_error(original, denoised)

def calculate_psnr(original, denoised):
    # Ensure they are the same type/range
    data_range = 255 if original.dtype == np.uint8 else 1.0
    return peak_signal_noise_ratio(original, denoised, data_range=data_range)

def calculate_ssim(original, denoised):
    is_multichannel = len(original.shape) == 3
    # Use win_size smaller than image if image is too small
    win_size = min(7, min(original.shape[:2])) 
    if win_size % 2 == 0:
        win_size -= 1
        
    if is_multichannel:
        return structural_similarity(original, denoised, channel_axis=2, data_range=255, win_size=win_size)
    else:
        return structural_similarity(original, denoised, data_range=255, win_size=win_size)
