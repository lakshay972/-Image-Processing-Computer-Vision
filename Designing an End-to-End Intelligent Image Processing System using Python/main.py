# Name         : Lakshay
# Roll No      : 2301010436
# Course       : Image Processing & Computer Vision
# Assignment   : Designing an End-to-End Intelligent Image Processing System using Python
# Date         : 13/04/2026

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import urllib.request
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import mean_squared_error as mse_metric

os.makedirs("outputs", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)

IMAGE_URLS = {
    "sample_images/image1.jpg": "https://images8.alphacoders.com/104/1041414.jpg",
    "sample_images/image2.jpg": "https://images.pexels.com/photos/45164/mare-animal-nature-ride-45164.jpeg?cs=srgb&dl=animal-white-mane-45164.jpg&fm=jpg",
    "sample_images/image3.jpg": "https://cdn.pixabay.com/photo/2018/02/18/13/10/nature-3162233_1280.jpg",
}

def download_images():
    print("Downloading sample images...")
    for path, url in IMAGE_URLS.items():
        if os.path.exists(path):
            print(" already exists:", path)
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r, open(path, 'wb') as f:
                f.write(r.read())
            print(" downloaded:", path)
        except Exception as e:
            print(" failed:", path, "->", str(e))


# Task 2 - Image Acquisition and Preprocessing

def task2_acquisition(image_path):
    print("\n[Task 2] Image Acquisition and Preprocessing")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Cannot load: " + image_path)

    resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    print("  original size :", image.shape[1], "x", image.shape[0])
    print("  resized to    : 512 x 512")
    print("  converted to grayscale")

    return resized, gray


# Task 3 - Image Enhancement and Restoration

def add_gaussian_noise(image, sigma=25):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def add_salt_pepper_noise(image, prob=0.02):
    noisy = image.copy()
    total = image.size
    num = int(total * prob / 2)
    coords = [np.random.randint(0, i, num) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, num) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def task3_enhancement(gray):
    print("\n[Task 3] Image Enhancement and Restoration")

    noisy_gaussian = add_gaussian_noise(gray)
    noisy_sp = add_salt_pepper_noise(gray)

    mean_filtered = cv2.blur(noisy_gaussian, (5, 5))
    median_filtered = cv2.medianBlur(noisy_sp, 5)
    gaussian_filtered = cv2.GaussianBlur(noisy_gaussian, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gaussian_filtered)

    print("  added gaussian noise and salt-pepper noise")
    print("  applied mean, median, gaussian filters")
    print("  applied CLAHE contrast enhancement")

    return noisy_gaussian, noisy_sp, mean_filtered, median_filtered, gaussian_filtered, enhanced


# Task 4 - Segmentation and Morphological Processing

def task4_segmentation(enhanced):
    print("\n[Task 4] Image Segmentation and Morphological Processing")

    _, global_thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    otsu_val, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(otsu_thresh, kernel, iterations=2)
    eroded = cv2.erode(otsu_thresh, kernel, iterations=2)

    print("  global threshold : 127")
    print("  otsu threshold   :", otsu_val)
    print("  dilation and erosion applied")

    return global_thresh, otsu_thresh, dilated, eroded


# Task 5 - Object Representation and Feature Extraction

def task5_features(enhanced, original_bgr):
    print("\n[Task 5] Object Representation and Feature Extraction")

    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
    canny = cv2.Canny(enhanced, 50, 150)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 300]

    canvas = original_bgr.copy()
    for cnt in contours[:20]:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    orb = cv2.ORB_create(nfeatures=500)
    keypoints, _ = orb.detectAndCompute(enhanced, None)
    kp_image = cv2.drawKeypoints(
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 200, 0)
    )

    print("  sobel and canny edges detected")
    print("  contours found:", len(contours))
    print("  ORB keypoints :", len(keypoints))

    return sobel, canny, canvas, kp_image


# Task 6 - Performance Evaluation

def task6_evaluation(gray, enhanced, gaussian_filtered, mean_filtered):
    print("\n[Task 6] Performance Evaluation")

    def evaluate(ref, target, label):
        mse = mse_metric(ref, target)
        psnr = psnr_metric(ref, target, data_range=255)
        s = ssim_metric(ref, target, data_range=255)
        print(f"  {label}")
        print(f"    MSE  : {mse:.4f}")
        print(f"    PSNR : {psnr:.4f} dB")
        print(f"    SSIM : {s:.4f}")
        return mse, psnr, s

    m1 = evaluate(gray, enhanced,         "Original vs Enhanced")
    m2 = evaluate(gray, gaussian_filtered, "Original vs Gaussian Filtered")
    m3 = evaluate(gray, mean_filtered,     "Original vs Mean Filtered")

    return m1, m2, m3


# Task 7 - Final Visualization

def task7_visualization(gray, noisy_gaussian, gaussian_filtered, enhanced,
                        otsu_thresh, kp_image, metrics, name):
    print("\n[Task 7] Final Visualization")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Intelligent Image Processing Pipeline - " + name, fontsize=13)

    panels = [
        (gray,              "1. Original Grayscale",      'gray'),
        (noisy_gaussian,    "2. Noisy (Gaussian)",        'gray'),
        (gaussian_filtered, "3. Restored (Gaussian Filter)",'gray'),
        (enhanced,          "4. Enhanced (CLAHE)",        'gray'),
        (otsu_thresh,       "5. Segmented (Otsu)",        'gray'),
        (kp_image,          "6. Features (ORB)",          None),
    ]

    for ax, (img, title, cmap) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    out = "outputs/" + os.path.splitext(name)[0] + "_pipeline.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print("  saved:", out)

    m1, m2, m3 = metrics
    print("\n  Summary:")
    print(f"    CLAHE Enhancement  -> PSNR={m1[1]:.2f} dB  SSIM={m1[2]:.4f}")
    print(f"    Gaussian Filter    -> PSNR={m2[1]:.2f} dB  SSIM={m2[2]:.4f}")
    print(f"    Mean Filter        -> PSNR={m3[1]:.2f} dB  SSIM={m3[2]:.4f}")
    print("  Higher PSNR and SSIM closer to 1.0 means better image quality.")


def process(image_path):
    name = os.path.basename(image_path)
    print("\n" + "=" * 50)
    print("Image:", name)
    print("=" * 50)

    original_bgr, gray = task2_acquisition(image_path)

    noisy_g, noisy_sp, mean_f, median_f, gauss_f, enhanced = task3_enhancement(gray)
    global_t, otsu_t, dilated, eroded = task4_segmentation(enhanced)
    sobel, canny, contour_canvas, kp_image = task5_features(enhanced, original_bgr)
    metrics = task6_evaluation(gray, enhanced, gauss_f, mean_f)
    task7_visualization(gray, noisy_g, gauss_f, enhanced, otsu_t, kp_image, metrics, name)

    base = os.path.splitext(name)[0]
    cv2.imwrite("outputs/" + base + "_original.png",  gray)
    cv2.imwrite("outputs/" + base + "_noisy.png",     noisy_g)
    cv2.imwrite("outputs/" + base + "_restored.png",  gauss_f)
    cv2.imwrite("outputs/" + base + "_enhanced.png",  enhanced)
    cv2.imwrite("outputs/" + base + "_segmented.png", otsu_t)
    cv2.imwrite("outputs/" + base + "_dilated.png",   dilated)
    cv2.imwrite("outputs/" + base + "_eroded.png",    eroded)
    cv2.imwrite("outputs/" + base + "_sobel.png",     sobel)
    cv2.imwrite("outputs/" + base + "_canny.png",     canny)
    cv2.imwrite("outputs/" + base + "_contours.png",  contour_canvas)
    cv2.imwrite("outputs/" + base + "_features.png",  kp_image)


def main():
    print("Intelligent Image Enhancement and Analysis System")
    print("-------------------------------------------------")

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        download_images()
        paths = list(IMAGE_URLS.keys())

    for p in paths:
        if os.path.exists(p):
            process(p)
        else:
            print("file not found:", p)

    print("\nDone. Check outputs/ folder.")


if __name__ == "__main__":
    main()
