import os
import cv2
from skimage import data

def download_datasets(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    images = {
        'camera': data.camera(), # grayscale
        'astronaut': cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR),
    }
    
    downloaded_files = []
    for name, img in images.items():
        filepath = os.path.join(data_dir, f"{name}.png")
        downloaded_files.append(filepath)
        if not os.path.exists(filepath):
            cv2.imwrite(filepath, img)
            
    return downloaded_files
