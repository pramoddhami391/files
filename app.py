"""
Part D: Combined Pipeline as a Streamlit App
Medicine Label Reader: photo -> English text (OCR) -> Nepali text (translation)
-> Nepali audio (TTS)

Run with:
    streamlit run app.py --server.address=0.0.0.0 --server.port=8501
"""

import streamlit as st
import easyocr
from deep_translator import GoogleTranslator
from gtts import gTTS
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import re


def preprocess_image(pil_image):
    """
    Enhances medicine packaging photos for better OCR recognition:
    - 2x Bicubic Upscaling (improves small font detection)
    - Grayscale conversion
    - CLAHE (Contrast Limited Adaptive Histogram Equalization) to reduce foil/metallic glare
    """
    img_np = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2x Bicubic scaling for small fonts
    h, w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Convert to Grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement for foil / shiny label glare
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    return contrast_enhanced


def clean_ocr_text(text):
    """Fix common OCR digit/letter misclassifications in medical dosages and terms."""
    if not text:
        return text
    # Fix digit '0' accidentally placed inside words (e.g., Paracetam0l -> Paracetamol)
    text = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'o', text)
    # Fix digit '1' placed inside words (e.g., Med1cine -> Medicine)
    text = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'l', text)
    # Ensure standard spacing between numbers and dosage units (e.g., 500mg -> 500 mg)
    text = re.sub(r'(\d+)\s*(mg|ml|mcg|g)\b', r'\1 \2', text, flags=re.IGNORECASE)
    return text


# Cache the OCR reader so it only loads once per session
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.title("Medicine Label Reader")
st.write("Upload a photo of a medicine strip or box to hear its details in Nepali.")

uploaded_file = st.file_uploader("Upload a medicine label", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Or capture from camera")

image_file = uploaded_file or camera_file

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Selected label", use_container_width=True)

    # Preprocess image for OCR
    processed_img = preprocess_image(image)

    with st.expander("🔍 View Preprocessed Image (Contrast & Scaling Enhanced)"):
        st.image(processed_img, caption="Preprocessed Image for EasyOCR", use_container_width=True)

    # --- Stage 1: OCR (vision) ---
    st.subheader("Extracted text")
    with st.spinner("Reading label..."):
        # Tuned EasyOCR parameters for small text and low-contrast packaging
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
        raw_text = " ".join(results)
        english_text = clean_ocr_text(raw_text)

    if not english_text.strip():
        st.warning("No text detected. Try a clearer, well-lit photo.")
    else:
        st.write("English:", english_text)

        # --- Stage 2: Translation (NLP) ---
        st.subheader("Nepali translation")
        with st.spinner("Translating..."):
            nepali_text = GoogleTranslator(source='en', target='ne').translate(english_text)
        st.write("Nepali:", nepali_text)

        # --- Stage 3: Text-to-speech ---
        st.subheader("Listen")
        with st.spinner("Generating audio..."):
            tts = gTTS(text=nepali_text, lang="ne")
            mp3_bytes = BytesIO()
            tts.write_to_fp(mp3_bytes)
            mp3_bytes.seek(0)
        st.audio(mp3_bytes.read(), format="audio/mp3")

