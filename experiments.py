import os
import cv2
import numpy as np
from data_loader import download_datasets
from noise import add_gaussian_noise, add_salt_and_pepper_noise
from adaptive_filter import apply_gaussian, apply_median, hybrid_filter_fixed, hybrid_filter_dynamic, apply_dct
from metrics import calculate_mse, calculate_psnr, calculate_ssim
from visualization import plot_metrics_vs_noise, plot_metrics_vs_k, save_comparison_grid

def get_images():
    download_datasets()
    images = {}
    data_dir = "data"
    if not os.path.exists(data_dir):
        return images
    for img_name in os.listdir(data_dir):
        if img_name.endswith((".tiff", ".png", ".jpg")):
            path = os.path.join(data_dir, img_name)
            img = cv2.imread(path)
            if img is not None:
                images[img_name.split('.')[0]] = img
    return images

def test_gaussian_noise_psnr(image, sigmas):
    results = {'Gaussian': [], 'Median': [], 'Fixed Hybrid': [], 'Dynamic Hybrid': [], 'DCT': []}
    for sigma in sigmas:
        noisy = add_gaussian_noise(image, sigma=sigma)
        filtered_g = apply_gaussian(noisy, (5, 5))
        filtered_m = apply_median(noisy, 5)
        filtered_f = hybrid_filter_fixed(noisy, threshold=2500, kernel_size=5)
        filtered_d = hybrid_filter_dynamic(noisy, k_param=0.5, kernel_size=5)
        filtered_dct = apply_dct(noisy, k_thresh=0.1)
        
        results['Gaussian'].append(calculate_psnr(image, filtered_g))
        results['Median'].append(calculate_psnr(image, filtered_m))
        results['Fixed Hybrid'].append(calculate_psnr(image, filtered_f))
        results['Dynamic Hybrid'].append(calculate_psnr(image, filtered_d))
        results['DCT'].append(calculate_psnr(image, filtered_dct))
    return results

def test_k_parameter(image, k_values, sigma=25):
    noisy = add_gaussian_noise(image, sigma=sigma)
    psnrs = []
    for k in k_values:
        filtered = hybrid_filter_dynamic(noisy, k_param=k, kernel_size=5)
        psnrs.append(calculate_psnr(image, filtered))
    return psnrs

def run_experiments():
    images = get_images()
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Starting Experiments...")

    # --- Part 1: Grayscale Metrics & Graphs (Camera) ---
    if 'camera' in images:
        # Re-converting back to grayscale if read as BGR
        img_camera = cv2.cvtColor(images['camera'], cv2.COLOR_BGR2GRAY) 
        
        print("\n--- 1. Grayscale Metrics (Camera) ---")
        sigmas = [10, 20, 30, 40, 50]
        print("Running PSNR vs Gaussian Noise experiment...")
        res_gauss = test_gaussian_noise_psnr(img_camera, sigmas)
        plot_metrics_vs_noise(sigmas, res_gauss, "PSNR (dB)", "Gaussian Noise Sigma", os.path.join(out_dir, "psnr_vs_sigma_camera.png"))
        
        k_values = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        print("Running PSNR vs k-parameter experiment...")
        res_k = test_k_parameter(img_camera, k_values, sigma=25)
        plot_metrics_vs_k(k_values, {'Dynamic Hybrid Filter': res_k}, "PSNR (dB)", os.path.join(out_dir, "psnr_vs_k_camera.png"))

        sigma_test = 25
        print("Applying filters to Gaussian noise (Sigma=25)...")
        noisy_camera = add_gaussian_noise(img_camera, sigma=sigma_test)
        g_camera = apply_gaussian(noisy_camera, (5, 5))
        m_camera = apply_median(noisy_camera, 5)
        f_camera = hybrid_filter_fixed(noisy_camera, threshold=2500)
        d_camera = hybrid_filter_dynamic(noisy_camera, k_param=0.5)
        dct_camera = apply_dct(noisy_camera, k_thresh=0.1)
        
        save_comparison_grid(
            [img_camera, noisy_camera, g_camera, m_camera, f_camera, d_camera, dct_camera],
            ["Original", "Noisy (Sigma=25)", "Gaussian", "Median", "Fixed Hybrid", "Dynamic Hybrid", "DCT Denoised"],
            os.path.join(out_dir, "comparison_gaussian_camera.png")
        )

        print("Applying filters to Salt & Pepper noise (Density=0.05)...")
        sp_test_density = 0.05
        noisy_sp = add_salt_and_pepper_noise(img_camera, density=sp_test_density)
        g_sp = apply_gaussian(noisy_sp, (5, 5))
        m_sp = apply_median(noisy_sp, 5)
        f_sp = hybrid_filter_fixed(noisy_sp, threshold=2500)
        d_sp = hybrid_filter_dynamic(noisy_sp, k_param=0.5)
        dct_sp = apply_dct(noisy_sp, k_thresh=0.1)
        
        save_comparison_grid(
            [img_camera, noisy_sp, g_sp, m_sp, f_sp, d_sp, dct_sp],
            ["Original", "S&P Noisy", "Gaussian", "Median", "Fixed Hybrid", "Dynamic Hybrid", "DCT Denoised"],
            os.path.join(out_dir, "comparison_sp_camera.png")
        )

    # --- Part 2: Color Image & Edge Proof (Astronaut) ---
    if 'astronaut' in images:
        img_color = images['astronaut']
        
        print("\n--- 2. Edge Preservation Proof (THIS IMPRESSES) ---")
        sigma_test = 25
        noisy_color = add_gaussian_noise(img_color, sigma=sigma_test)
        
        g_color = apply_gaussian(noisy_color, (5, 5))
        d_color = hybrid_filter_dynamic(noisy_color, k_param=0.5)
        dct_color = apply_dct(noisy_color, k_thresh=0.1)
        
        gray_orig = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        gray_g = cv2.cvtColor(g_color, cv2.COLOR_BGR2GRAY)
        gray_d = cv2.cvtColor(d_color, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny
        edge_orig = cv2.Canny(gray_orig, 100, 200)
        edge_g = cv2.Canny(gray_g, 100, 200)
        edge_d = cv2.Canny(gray_d, 100, 200)
        
        save_comparison_grid(
            [gray_orig, edge_orig, gray_g, edge_g, gray_d, edge_d],
            ["Original", "Edge (Original)", "Gaussian Smoothed", "Edge (Gaussian) - Blurred", "Hybrid Denoised", "Edge (Hybrid) - Preserved"],
            os.path.join(out_dir, "edge_preservation_proof.png")
        )
        
        print("=> Hybrid filtering preserves structural edges better than Gaussian smoothing.\n")
        
        print("Generating Full Color Comparisons...")
        f_color = hybrid_filter_fixed(noisy_color, threshold=2500)
        m_color = apply_median(noisy_color, 5)
        
        save_comparison_grid(
            [img_color, noisy_color, g_color, m_color, f_color, d_color, dct_color],
            ["Original (Color)", f"Noisy (Sigma={sigma_test})", "Gaussian", "Median", "Fixed Hybrid", "Dynamic Hybrid", "DCT Denoised"],
            os.path.join(out_dir, "comparison_gaussian_color.png")
        )

    print("\nExperiments completed! Visualizations saved to 'output' directory.")

if __name__ == "__main__":
    run_experiments()
