"""
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
    """Enhance medicine packaging photos for better OCR recognition."""
    img_np = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def clean_ocr_text(text):
    """Fix common digit/letter misclassifications in medical terms."""
    if not text:
        return ""
    text = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'o', text)
    text = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'l', text)
    text = re.sub(r'(\d+)\s*(mg|ml|mcg|g)\b', r'\1 \2', text, flags=re.IGNORECASE)
    return text.strip()


def is_valid_ocr_fragment(text):
    """Filter out watermarks, barcode noise, and unreadable gibberish."""
    text_clean = text.strip()
    if len(text_clean) < 2:
        return False
    watermarks = ['shutterstock', 'stock', 'photo', 'vector', 'adobe', 'depositphotos', 'alamy']
    if any(w in text_clean.lower() for w in watermarks):
        return False
    if re.match(r'^\d{8,}$', text_clean):
        return False
    if re.search(r'[@"~`$^&*={}\[\]\\]', text_clean):
        return False
    return True


def filter_ocr_results(raw_ocr_output, min_confidence=0.45):
    clean_fragments = []
    for item in raw_ocr_output:
        bbox, text, score = item
        if score >= min_confidence and is_valid_ocr_fragment(text):
            cleaned = clean_ocr_text(text)
            if cleaned:
                clean_fragments.append(cleaned)
    return clean_fragments


# Cache local EasyOCR reader (loads lazily on CPU)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)


st.set_page_config(page_title="Medicine Label Reader", page_icon="💊")

st.title("Medicine Label Reader")
st.write("Upload a photo of a medicine strip or box to hear its details in Nepali.")

uploaded_file = st.file_uploader("Upload a medicine label", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Or capture from camera")

image_file = uploaded_file or camera_file

if image_file:
    image = Image.open(image_file).convert("RGB")
    try:
        st.image(image, caption="Selected label", use_container_width=True)
    except TypeError:
        st.image(image, caption="Selected label")


    # --- Stage 1: OCR (vision) ---
    st.subheader("Extracted text")
    with st.spinner("Reading label..."):
        processed_img = preprocess_image(image)
        reader = load_reader()
        raw_results = reader.readtext(
            processed_img,
            detail=1,
            mag_ratio=2.0,
            adjust_contrast=0.5
        )
        clean_fragments = filter_ocr_results(raw_results, min_confidence=0.45)
        english_text = " ".join(clean_fragments)

    if not english_text.strip():
        st.warning("No text detected. Try a clearer, well-lit photo.")
    else:
        st.write("English:", english_text)

        # --- Stage 2: Translation (NLP) ---
        st.subheader("Nepali translation")
        with st.spinner("Translating..."):
            nepali_translations = [GoogleTranslator(source='en', target='ne').translate(item) for item in clean_fragments]
            nepali_text = " ".join(nepali_translations)
        st.write("Nepali:", nepali_text)

        # --- Stage 3: Text-to-speech ---
        st.subheader("Listen")
        with st.spinner("Generating audio..."):
            speech_text = ". ".join(nepali_translations)
            tts = gTTS(text=speech_text, lang="ne")
            mp3_bytes = BytesIO()
            tts.write_to_fp(mp3_bytes)
            mp3_bytes.seek(0)
        st.audio(mp3_bytes.read(), format="audio/mp3")




