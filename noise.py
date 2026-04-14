import numpy as np
import cv2

def add_gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape[:2]
    if len(image.shape) == 3:
        ch = image.shape[2]
        gauss = np.random.normal(mean, sigma, (row, col, ch))
    else:
        gauss = np.random.normal(mean, sigma, (row, col))
        
    gauss = gauss.reshape(image.shape)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_and_pepper_noise(image, density=0.04):
    noisy = np.copy(image)
    salt_prob = density / 2
    pepper_prob = density / 2
    
    amount_salt = np.ceil(salt_prob * image.size / (image.shape[2] if len(image.shape) == 3 else 1))
    amount_pepper = np.ceil(pepper_prob * image.size / (image.shape[2] if len(image.shape) == 3 else 1))
    
    coords = [np.random.randint(0, i - 1, int(amount_salt)) for i in image.shape[:2]]
    if len(image.shape) == 3:
        for c in range(image.shape[2]):
            noisy[tuple(coords) + (c,)] = 255
    else:
        noisy[tuple(coords)] = 255

    coords = [np.random.randint(0, i - 1, int(amount_pepper)) for i in image.shape[:2]]
    if len(image.shape) == 3:
        for c in range(image.shape[2]):
            noisy[tuple(coords) + (c,)] = 0
    else:
        noisy[tuple(coords)] = 0
        
    return noisy
