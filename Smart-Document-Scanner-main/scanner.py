"""
------------------------------------------------------------
Name: Lakshay
Roll No: 2301010436
Course: Image Processing & Computer Vision
Unit: Image Acquisition, Sampling & Quantization
Assignment Title: Smart Document Scanner & Quality Analysis System
Date: 12-Feb-2026
------------------------------------------------------------
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("====================================================")
print(" SMART DOCUMENT SCANNER & QUALITY ANALYSIS SYSTEM ")
print("====================================================")
print("This system simulates document digitization and")
print("analyzes the effects of sampling and quantization.")
print("----------------------------------------------------")

if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Task 2: Image Acquisition

image_path = input("Enter document image path: ")
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Resize to 512x512
image = cv2.resize(image, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.png", image)
cv2.imwrite("outputs/grayscale.png", gray)

# Task 3: Sampling

def sample_image(img, size):
    small = cv2.resize(img, (size, size))
    upscaled = cv2.resize(small, (512, 512))
    return upscaled

high_res = gray  # 512x512
medium_res = sample_image(gray, 256)
low_res = sample_image(gray, 128)

cv2.imwrite("outputs/sample_512.png", high_res)
cv2.imwrite("outputs/sample_256.png", medium_res)
cv2.imwrite("outputs/sample_128.png", low_res)

# Task 4: Quantization

def quantize_image(img, levels):
    step = 256 // levels
    quantized = (img // step) * step
    return quantized

quant_8bit = gray  
quant_4bit = quantize_image(gray, 16)
quant_2bit = quantize_image(gray, 4)

cv2.imwrite("outputs/quant_8bit.png", quant_8bit)
cv2.imwrite("outputs/quant_4bit.png", quant_4bit)
cv2.imwrite("outputs/quant_2bit.png", quant_2bit)

# Task 5: Comparison Figure

plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.title("512x512")
plt.imshow(high_res, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.title("256x256")
plt.imshow(medium_res, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.title("128x128")
plt.imshow(low_res, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.title("8-bit")
plt.imshow(quant_8bit, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.title("4-bit")
plt.imshow(quant_4bit, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.title("2-bit")
plt.imshow(quant_2bit, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/comparison_figure.png")
plt.show()

# Observations

print("\n===== QUALITY OBSERVATIONS =====")

print("\n1. Sampling Analysis:")
print("- 512x512 retains full clarity and sharp edges.")
print("- 256x256 slightly reduces fine text detail.")
print("- 128x128 causes blurred edges and text distortion.")
print("- Lower resolution significantly affects OCR accuracy.")

print("\n2. Quantization Analysis:")
print("- 8-bit (256 levels) maintains natural grayscale appearance.")
print("- 4-bit (16 levels) introduces visible banding.")
print("- 2-bit (4 levels) causes severe information loss.")
print("- Low bit-depth reduces readability and OCR performance.")

print("\n3. OCR Suitability:")
print("- High resolution + 8-bit grayscale is best for OCR.")
print("- Low resolution + 2-bit quantization is unsuitable.")
print("- Proper acquisition improves digitization quality.")

print("\nAll outputs saved inside 'outputs/' folder.")
print("Assignment Completed Successfully!")
