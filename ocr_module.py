"""
Part A: Text Extraction from Medicine Label (Vision / OCR)
Standalone test script - run this first on its own before wiring into the app.

Install:
    pip install easyocr pillow
"""

import easyocr

# Step 1: Load the OCR reader (English only, since medicine labels are usually
# printed in English even in Nepal)
reader = easyocr.Reader(['en'])

# Step 2: Point this at a photo of a medicine strip/box
IMAGE_PATH = "tablet_medicine.jpg"

# Step 3: Run OCR. detail=0 returns just the text strings, no bounding boxes
results = reader.readtext(IMAGE_PATH, detail=0)

# Step 4: Join the detected text fragments into one string
extracted_text = " ".join(results)

print("Raw OCR fragments:", results)
print("Extracted Text:", extracted_text)
