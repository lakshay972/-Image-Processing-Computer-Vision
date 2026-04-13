# Name         : Lakshay
# Roll No      : 2301010436
# Course       : Image Processing & Computer Vision
# Assignment   : Compression and Segmentation of Medical Images using Python
# Date         : 13-04-26

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from itertools import groupby

os.makedirs("outputs", exist_ok=True)


# Task 1 - Run Length Encoding compression

def rle_encode(image):
    flat = image.flatten()
    encoded = []
    for val, group in groupby(flat):
        encoded.append((val, len(list(group))))
    return encoded

def rle_decode(encoded, shape):
    flat = []
    for val, count in encoded:
        flat.extend([val] * count)
    return np.array(flat, dtype=np.uint8).reshape(shape)

def compression_ratio(image, encoded):
    original_size = image.size
    encoded_size = len(encoded) * 2
    ratio = original_size / encoded_size
    savings = (1 - encoded_size / original_size) * 100
    return ratio, savings


# Task 2 - Segmentation using thresholding

def apply_thresholding(image):
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    otsu_val, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return global_thresh, otsu_thresh, otsu_val


# Task 3 - Morphological operations

def apply_morphology(segmented):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(segmented, kernel, iterations=2)
    eroded = cv2.erode(segmented, kernel, iterations=2)
    return dilated, eroded


# Task 4 - Save comparison figure

def save_comparison(image, global_thresh, otsu_thresh, dilated, eroded, ratio, savings, name):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Medical Image Processing - " + name, fontsize=13)

    axes[0][0].imshow(image, cmap='gray')
    axes[0][0].set_title("Original")
    axes[0][0].axis('off')

    axes[0][1].imshow(global_thresh, cmap='gray')
    axes[0][1].set_title("Global Thresholding (T=127)")
    axes[0][1].axis('off')

    axes[0][2].imshow(otsu_thresh, cmap='gray')
    axes[0][2].set_title("Otsu Thresholding")
    axes[0][2].axis('off')

    axes[1][0].imshow(dilated, cmap='gray')
    axes[1][0].set_title("Dilation")
    axes[1][0].axis('off')

    axes[1][1].imshow(eroded, cmap='gray')
    axes[1][1].set_title("Erosion")
    axes[1][1].axis('off')

    axes[1][2].axis('off')
    axes[1][2].text(0.1, 0.6,
        f"Compression Ratio: {ratio:.4f}\nStorage Savings: {savings:.2f}%\n\n"
        "Otsu thresholding performs better\nthan global for medical images\n"
        "as it adapts to image histogram.\n\n"
        "Dilation fills small gaps.\nErosion removes noise pixels.",
        fontsize=9, verticalalignment='top', transform=axes[1][2].transAxes)

    plt.tight_layout()
    out = "outputs/" + os.path.splitext(name)[0] + "_result.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print("saved:", out)


def process(image_path):
    name = os.path.basename(image_path)
    print("\nProcessing:", name)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("could not load:", image_path)
        return

    # Task 1
    encoded = rle_encode(image)
    ratio, savings = compression_ratio(image, encoded)
    decoded = rle_decode(encoded, image.shape)
    lossless_ok = np.array_equal(image, decoded)

    print("  Original size  :", image.size, "pixels")
    print("  Compression ratio:", round(ratio, 4))
    print("  Storage savings  :", round(savings, 2), "%")
    print("  Lossless check   :", lossless_ok)

    # Task 2
    global_thresh, otsu_thresh, otsu_val = apply_thresholding(image)
    print("  Otsu threshold value:", otsu_val)

    cv2.imwrite("outputs/" + os.path.splitext(name)[0] + "_global.png", global_thresh)
    cv2.imwrite("outputs/" + os.path.splitext(name)[0] + "_otsu.png", otsu_thresh)

    # Task 3
    dilated, eroded = apply_morphology(otsu_thresh)
    cv2.imwrite("outputs/" + os.path.splitext(name)[0] + "_dilated.png", dilated)
    cv2.imwrite("outputs/" + os.path.splitext(name)[0] + "_eroded.png", eroded)

    # Task 4
    save_comparison(image, global_thresh, otsu_thresh, dilated, eroded, ratio, savings, name)


def main():
    print("Medical Image Compression and Segmentation System")
    print("--------------------------------------------------")

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = [
            "Medical/sample_images/xray.webp",
            "Medical/sample_images/mri.webp",
            "Medical/sample_images/ct.jpg"
        ]

    for p in paths:
        if os.path.exists(p):
            process(p)
        else:
            print("file not found:", p)

    print("\nDone. Check outputs/ folder.")


if __name__ == "__main__":
    main()
