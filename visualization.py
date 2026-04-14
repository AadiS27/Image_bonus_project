import matplotlib.pyplot as plt
import cv2

def plot_metrics_vs_noise(noise_levels, metric_dict, metric_name, x_label, save_path):
    plt.figure(figsize=(10, 6))
    for method, values in metric_dict.items():
        plt.plot(noise_levels, values, marker='o', label=method)
    plt.title(f"{metric_name} vs {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics_vs_k(k_values, metric_dict, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    for label, values in metric_dict.items():
        plt.plot(k_values, values, marker='s', label=label)
    plt.title(f"{metric_name} vs Threshold Parameter (k)")
    plt.xlabel("k parameter")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_comparison_grid(images, labels, save_path):
    num_images = len(images)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_images == 1:
        axes = [axes]
    elif rows > 1:
        axes = axes.flatten()
        
    for i in range(len(axes)):
        if i < num_images:
            img = images[i]
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
            else:
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(labels[i])
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
