# Name         : Lakshay
# Roll No      : 2301010436
# Course       : Image Processing & Computer Vision
# Assignment   : Object Representation and Feature Extraction for Traffic Images
# Date         : 13/04/26

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import urllib.request

os.makedirs("outputs", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)

# public domain traffic images from wikimedia
IMAGE_URLS = {
    "sample_images/traffic1.jpg": "https://imgk.timesnownews.com/story/1563956019-Traffic_representative_img.jpg",
    "sample_images/traffic2.jpg": "https://cdn.pixabay.com/photo/2022/05/22/11/10/highway-7213206_1280.jpg",
    "sample_images/traffic3.jpg": "https://cdn.pixabay.com/photo/2021/11/20/05/15/car-6810885_1280.jpg",
}

def download_images():
    print("Downloading sample traffic images...")
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


# Task 1 - Edge Detection

def task1_edge_detection(gray, name):
    print("  Task 1: Edge Detection")

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

    canny = cv2.Canny(gray, 50, 150)

    base = os.path.splitext(name)[0]
    cv2.imwrite("outputs/" + base + "_sobel.png", sobel)
    cv2.imwrite("outputs/" + base + "_canny.png", canny)

    print("    Sobel edge pixels:", np.sum(sobel > 30))
    print("    Canny edge pixels:", np.sum(canny > 0))

    return sobel, canny


# Task 2 - Object Representation

def task2_object_representation(gray, original, name):
    print("  Task 2: Object Representation")

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 500]

    canvas = original.copy()
    for i, cnt in enumerate(contours[:15]):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"    Object {i+1}: area={area:.1f}  perimeter={perimeter:.1f}")

    base = os.path.splitext(name)[0]
    cv2.imwrite("outputs/" + base + "_contours.png", canvas)
    print("    Total objects detected:", len(contours))

    return contours, canvas


# Task 3 - Feature Extraction using ORB

def task3_feature_extraction(gray, name):
    print("  Task 3: Feature Extraction (ORB)")

    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    kp_image = cv2.drawKeypoints(
        gray, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )

    base = os.path.splitext(name)[0]
    cv2.imwrite("outputs/" + base + "_orb.png", kp_image)
    print("    Keypoints found:", len(keypoints))

    return keypoints, descriptors, kp_image


# Task 4 - Comparative Analysis

def task4_analysis(gray, sobel, canny, contour_canvas, kp_image, name):
    print("  Task 4: Comparative Analysis")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Traffic Monitoring System - " + name, fontsize=13)

    axes[0][0].imshow(gray, cmap='gray')
    axes[0][0].set_title("Original Grayscale")
    axes[0][0].axis('off')

    axes[0][1].imshow(sobel, cmap='gray')
    axes[0][1].set_title("Sobel Edge Detection")
    axes[0][1].axis('off')

    axes[0][2].imshow(canny, cmap='gray')
    axes[0][2].set_title("Canny Edge Detection")
    axes[0][2].axis('off')

    axes[1][0].imshow(cv2.cvtColor(contour_canvas, cv2.COLOR_BGR2RGB))
    axes[1][0].set_title("Contours and Bounding Boxes")
    axes[1][0].axis('off')

    axes[1][1].imshow(cv2.cvtColor(kp_image, cv2.COLOR_BGR2RGB))
    axes[1][1].set_title("ORB Keypoints")
    axes[1][1].axis('off')

    axes[1][2].axis('off')
    axes[1][2].text(0.05, 0.9,
        "Comparison:\n\n"
        "Sobel:\n"
        " - Computes gradient magnitude\n"
        " - Thicker, noisier edges\n"
        " - Faster to compute\n\n"
        "Canny:\n"
        " - Multi-stage pipeline\n"
        " - Thin, clean edges\n"
        " - Better for detection tasks\n\n"
        "ORB:\n"
        " - Scale and rotation invariant\n"
        " - Good for vehicle matching\n"
        " - Works in real-time",
        fontsize=8.5, verticalalignment='top', transform=axes[1][2].transAxes)

    plt.tight_layout()
    out = "outputs/" + os.path.splitext(name)[0] + "_result.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print("  saved:", out)


def process(image_path):
    name = os.path.basename(image_path)
    print("\nProcessing:", name)

    original = cv2.imread(image_path)
    if original is None:
        print("  could not load:", image_path)
        return

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    sobel, canny = task1_edge_detection(gray, name)
    contours, contour_canvas = task2_object_representation(gray, original, name)
    keypoints, descs, kp_image = task3_feature_extraction(gray, name)
    task4_analysis(gray, sobel, canny, contour_canvas, kp_image, name)


def main():
    print("Feature-Based Traffic Monitoring System")
    print("----------------------------------------")

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
