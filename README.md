# Adaptive Image Denoising Using Hybrid Filters with Dynamic Thresholding

## About the Project
This project explores an advanced approach to digital image denoising by combining multiple traditional filtering techniques into a single, content-aware system. It dynamically assesses local image statistics to intelligently apply the optimal filtering technique to specific regions of an image.

This work demonstrates how mathematical image processing can move beyond blanket algorithmic applications (like blurring an entire image) to dynamic, structural-aware filtering that preserves vital high-frequency data—bridging the gap between pure noise suppression and edge preservation. 

## Team Information
**Course:** Image Processing and Analysis (IPA) - Bonus Work Submission

**Team Members:**
* Aadi Singhal (202352301)
* Tanay Komawar (202351146)
* Sudeep Gupta (202352334)

## Motivation
Traditional image processing relies heavily on static, linear and non-linear filters. While these techniques are foundational, they inherently possess severe mathematical limitations when applied globally:

* **Gaussian Smoothing:** Effectively dilutes continuous noise (like Gaussian distributions) but mathematically obliterates high-frequency data, causing severe structural edge blurring.
* **Median Filtering:** Excellent for isolating and removing impulse noise (Salt & Pepper), but highly aggressive and can cause clustering in flat/smooth gradients.

A **Dynamic Hybrid Filter** was chosen for this bonus work because it represents a major conceptual advancement. Instead of applying a black-box API, this project:
1. **Requires manual implementation** of complex algorithms (Manual Convolutions, Vectorized Sliding Windows, and Discrete Matrix Transforms).
2. **Introduces intelligent thresholding** calculated recursively.
3. **Moves from static processing** to region-aware analytical processing.

This project allowed us to deeply explore how statistical variance correlates to spatial frequency, extending our knowledge conceptually and programmatically beyond the core techniques taught in the IPA course.

## Objectives
* Understand and implement **2D Gaussian Convolution** from absolute scratch using generated probability matrices.
* Implement **Vectorized Median Filtering** applying sliding windows through spatial arrays and hardware-level sorting.
* Learn how **local and global statistical variance** can dictate the presence of edges versus noise.
* Prove mathematically and visually that dynamic thresholding preserves high-frequency structural edges.
* Study **Discrete Cosine Transformations (DCT)** for block-based frequency domain smoothing using self-generated transform matrices.

## Key Concepts Explained

### 1. Manual Gaussian Filtering
A linear spatial filter. Instead of yielding to library abstractions, we dynamically construct a 2D Gaussian probability distribution function matrix, normalize it, and explicitly convolve it layer-by-layer across the spatial domain to dampen high-frequency statistical noise.

### 2. Manual Median Filtering
A non-linear spatial filter. We extract a multi-dimensional array of spatial blocks using a strided sliding window. We then perform a hardware-level sort across the flattened spatial neighborhood and extract the exact median integer. Highly effective against Salt & Pepper (impulse) noise.

### 3. Dynamic Hybrid Thresholding
Our core conceptual algorithm. It computes a localized variance matrix by taking the difference between the mean of squares and the square of means across the image. It sets a global noise threshold using `k * global_variance`. 
* **High variance (Edges)** bypasses the blur and routes to the median filter.
* **Low variance (Smooth)** routes to the Gaussian filter.

### 4. Mathematical DCT Denoising
We manually construct an 8x8 basis Matrix (`C`). For every spatial 8x8 block in the image, we project it to the frequency domain via pure linear algebra `C @ Block @ C.T`. We threshold and drop weak frequency coefficients, and inverse transform it back `C.T @ DCT @ C`. DCT is computed using the standard transformation formula, but the coefficient thresholding is implemented entirely manually.

## Experimental Pipeline

The testing pipeline follows these strict algorithmic steps:

1. **Noise Injection**: Sample images are systematically injected with configurable standard deviations of Gaussian Noise and varying densities of Salt & Pepper noise.
2. **Filtering Execution**: The noisy matrices are passed in parallel through:
   * Pure Gaussian
   * Pure Median
   * The Dynamic Hybrid Model
   * The Manual DCT block-thresholder
3. **Mathematical Evaluation**: The structural outputs are computationally evaluated against the pristine ground-truth using PSNR (Peak-Signal-to-Noise Ratio), MSE (Mean Squared Error), and SSIM (Structural Similarity Index).
4. **Edge Preservation Proof**: The processed matrices are run through a Canny Edge detector alongside the original image to empirically validate structural retention.

## Results and Learnings

Through this project, we gained invaluable mathematical and programmatic insights:

* **Edge Preservation Proof:** Visually mapping the Canny edges proved definitively that *"Hybrid filtering preserves structural edges drastically better than Gaussian smoothing."*
* **The Mathematics of Convolution:** We realized exactly how border reflections, strides, and padding interact with matrix multiplication during spatial filtering.
* **Frequency vs Spatial:** We learned the distinction between operating in the raw spatial domain (Hybrid filters) compared to manipulating coefficients in the frequency domain (DCT).
* **Optimization Matters:** Implementing the sliding window median filter demonstrated that naive nested loops in python are unacceptably slow for image processing, forcing us to explore vectorized NumPy strides for instantaneous execution.

## Relevance to Image Processing and Analysis (IPA)

This project extends traditional IPA concepts in highly meaningful ways:

### From Global Application to Region-Aware Processing
* **Traditional IPA:** We learn about edge detection, filtering, and color spaces as separate, global operations.
* **Hybrid Filter:** We combine spatial statistics (variance) with non-linear filters to make localized, pixel-by-pixel filtering decisions.

### True Algorithmic Foundation
* Moving away from `cv2.GaussianBlur` and `cv2.medianBlur` proves a much deeper engagement with the coursework. Constructing convolution kernels, sorting sliding windows, and executing matrix multiplications for DCT proves an end-to-end understanding of what happens "under the hood" of standard Computer Vision pipelines.

### Modern Analytical Evaluation
* Validating image quality visually is subjective. Implementing algorithms to trace PSNR / SSIM curves against a dynamic parameter `$k$` demonstrates how to analytically tune filters for optimal structural results.

## Future Work
While this project focuses on foundational mathematical implementations, several extensions could bridge this approach with modern deep learning:
* **CNN Integration:** Compare our analytical hybrid filters directly against pre-trained Convolutional Neural Networks (like DnCNN) to measure computational trade-offs against AI-based denoising.
* **Learnable Thresholds ($k$):** Currently, the $k$ parameter is empirically chosen. Future work could implement a lightweight neural network to dynamically optimize the parameter $k$ based on local statistical textures.
* **Real-time Processing:** Porting the manual NumPy convolution and vectorized window operations to CUDA/GPU architectures for real-time video stream filtering.

## Conclusion

Key takeaways from this bonus project:
* **Hybrid models successfully bridge** the gap between impulse noise removal and continuous frequency smoothing without compromising structural density.
* **Using localized variance** as a binary spatial mask proves that statistical properties correlate directly to human recognizable features (edges).
* **Pure mathematical implementation** of Spatial Convolution, Non-Linear sliding windows, and Discrete frequency transformations severely deepens the fundamental understanding of digital image structure.

This exploration has significantly deepened our understanding of numerical matrices and spatial mathematics, beautifully complementing the foundational techniques learned in the IPA course.
