"""
------------------------------------------------------------
Name: Lakshay
Roll No: 2301010436
Course: Image Processing & Computer Vision
Unit: Noise Modeling & Image Restoration
Assignment Title: Noise Modeling and Image Restoration using Python
Date: 14-Feb-2026
------------------------------------------------------------
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

print("===================================================")
print(" IMAGE RESTORATION FOR SURVEILLANCE SYSTEMS ")
print("===================================================")

if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Task 1: Load Image

image_path = input("Enter surveillance image path: ")
image = cv2.imread(image_path)

if image is None:
    print("Error loading image.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.png", gray)

# Task 2: Noise Modeling

# Gaussian Noise
mean = 0
sigma = 25
gaussian_noise = np.random.normal(mean, sigma, gray.shape)
gaussian_noisy = gray + gaussian_noise
gaussian_noisy = np.clip(gaussian_noisy, 0, 255).astype(np.uint8)

cv2.imwrite("outputs/gaussian_noise.png", gaussian_noisy)

# Salt & Pepper Noise
sp_noisy = gray.copy()
prob = 0.02

num_salt = np.ceil(prob * gray.size * 0.5)
num_pepper = np.ceil(prob * gray.size * 0.5)

coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in gray.shape]
coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in gray.shape]

sp_noisy[tuple(coords_salt)] = 255
sp_noisy[tuple(coords_pepper)] = 0

cv2.imwrite("outputs/salt_pepper_noise.png", sp_noisy)


# Task 3: Filtering

def mean_filter(img):
    return cv2.blur(img, (3,3))

def median_filter(img):
    return cv2.medianBlur(img, 3)

def gaussian_filter(img):
    return cv2.GaussianBlur(img, (3,3), 0)

mean_g = mean_filter(gaussian_noisy)
median_g = median_filter(gaussian_noisy)
gauss_g = gaussian_filter(gaussian_noisy)

mean_sp = mean_filter(sp_noisy)
median_sp = median_filter(sp_noisy)
gauss_sp = gaussian_filter(sp_noisy)

cv2.imwrite("outputs/mean_gaussian.png", mean_g)
cv2.imwrite("outputs/median_gaussian.png", median_g)
cv2.imwrite("outputs/gaussian_gaussian.png", gauss_g)

cv2.imwrite("outputs/mean_sp.png", mean_sp)
cv2.imwrite("outputs/median_sp.png", median_sp)
cv2.imwrite("outputs/gaussian_sp.png", gauss_sp)

# Task 4: Performance Metrics

def mse(original, restored):
    return np.mean((original - restored) ** 2)

def psnr(original, restored):
    m = mse(original, restored)
    if m == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(m))

print("\n===== PERFORMANCE METRICS =====")

filters = {
    "Mean (Gaussian Noise)": mean_g,
    "Median (Gaussian Noise)": median_g,
    "Gaussian (Gaussian Noise)": gauss_g,
    "Mean (Salt & Pepper)": mean_sp,
    "Median (Salt & Pepper)": median_sp,
    "Gaussian (Salt & Pepper)": gauss_sp
}

for name, img in filters.items():
    print(f"\n{name}")
    print("MSE:", mse(gray, img))
    print("PSNR:", psnr(gray, img))


# Task 5: Analytical Discussion

print("\n===== ANALYSIS =====")
print("1. For Gaussian Noise:")
print("   - Gaussian filter performs best due to similar distribution.")
print("   - Mean filter reduces noise but blurs edges.")
print("   - Median filter moderately effective.")

print("\n2. For Salt & Pepper Noise:")
print("   - Median filter performs best.")
print("   - Mean filter fails to remove impulse noise.")
print("   - Gaussian filter less effective.")

print("\nBest Methods:")
print("- Gaussian noise → Gaussian filter")
print("- Salt & Pepper noise → Median filter")

print("\nAll output images saved in 'outputs/' folder.")
