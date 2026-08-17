"""
Part A: Text Extraction from Medicine Label (Vision / OCR)
Standalone test script - run this first on its own before wiring into the app.

Install:
    pip install easyocr pillow opencv-python-headless
"""

import easyocr
import cv2
import re
import os

from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def preprocess_image_path(image_path):
    """Enhance medicine packaging image using bicubic scaling and CLAHE contrast enhancement."""
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def clean_ocr_text(text):
    if not text:
        return text
    text = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'o', text)
    text = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'l', text)
    text = re.sub(r'(\d+)\s*(mg|ml|mcg|g)\b', r'\1 \2', text, flags=re.IGNORECASE)
    return text

# Step 1: Load the OCR reader
reader = easyocr.Reader(['en'])

# Step 2: Point this at a photo of a medicine strip/box
IMAGE_PATH = os.path.join(BASE_DIR, "medicine_label.jpg")



# Step 3: Preprocess image
processed_img = preprocess_image_path(IMAGE_PATH)

# Step 4: Run OCR with tuned sensitivity settings
results = reader.readtext(
    processed_img,
    detail=0,
    mag_ratio=2.0,
    text_threshold=0.4,
    low_text=0.3,
    link_threshold=0.4,
    contrast_ths=0.1,
    adjust_contrast=0.5
)

# Step 5: Clean and join detected text
raw_text = " ".join(results)
extracted_text = clean_ocr_text(raw_text)

print("Raw OCR fragments:", results)
print("Extracted Text:", extracted_text)

