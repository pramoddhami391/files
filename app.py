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
    return text.strip()


def is_valid_ocr_fragment(text):
    """Filter out watermarks, barcode noise, and unreadable gibberish."""
    text_clean = text.strip()
    if len(text_clean) < 2:
        return False
    
    # Filter stock photo watermarks
    watermarks = ['shutterstock', 'stock', 'photo', 'vector', 'adobe', 'depositphotos', 'alamy']
    if any(w in text_clean.lower() for w in watermarks):
        return False

    # Filter pure long barcode numbers (> 7 digits)
    if re.match(r'^\d{8,}$', text_clean):
        return False

    # Filter strings with weird symbols like @, ", ~, `
    if re.search(r'[@"~`$^&*={}\[\]\\]', text_clean):
        return False

    return True


def filter_ocr_results(raw_ocr_output, min_confidence=0.45):
    """
    Extracts high-confidence text fragments and filters out noise artifacts.
    Returns a list of clean text strings and structured tuples (text, score).
    """
    clean_fragments = []
    scored_fragments = []
    for item in raw_ocr_output:
        bbox, text, score = item
        if score >= min_confidence and is_valid_ocr_fragment(text):
            cleaned = clean_ocr_text(text)
            if cleaned:
                clean_fragments.append(cleaned)
                scored_fragments.append((cleaned, score))
    return clean_fragments, scored_fragments


# Cache the OCR reader so it only loads once per session
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.title("Medicine Label Reader")
st.write("Upload a photo of a medicine strip or box to hear its details in Nepali.")

# Sidebar Settings
st.sidebar.header("⚙️ OCR Tuning Controls")
min_confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
    help="Higher values remove background noise and gibberish; lower values include fainter text."
)

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
        # Run EasyOCR with detail=1 to get confidence scores per fragment
        raw_results = reader.readtext(
            processed_img,
            detail=1,
            mag_ratio=2.0,
            adjust_contrast=0.5
        )
        clean_fragments, scored_fragments = filter_ocr_results(raw_results, min_confidence=min_confidence)
        english_text = " | ".join(clean_fragments)

    if not clean_fragments:
        st.warning("No clear text detected above confidence threshold. Try lowering the Confidence Threshold in the sidebar or using a clearer photo.")
    else:
        st.markdown(f"**Extracted Text ({len(clean_fragments)} key items):**")
        for txt, score in scored_fragments:
            st.markdown(f"- **{txt}** *(confidence: {score:.0%})*")

        # --- Stage 2: Translation (NLP) ---
        st.subheader("Nepali translation")
        with st.spinner("Translating..."):
            nepali_translations = []
            for item in clean_fragments:
                translated_item = GoogleTranslator(source='en', target='ne').translate(item)
                nepali_translations.append(translated_item)
            
            nepali_text_full = " | ".join(nepali_translations)

        st.markdown("**Nepali Details:**")
        for orig, trans in zip(clean_fragments, nepali_translations):
            st.markdown(f"- **{trans}** *({orig})*")

        # --- Stage 3: Text-to-speech ---
        st.subheader("Listen")
        with st.spinner("Generating audio..."):
            # Concatenate Nepali text with clean pauses for natural TTS speech
            speech_text = ". ".join(nepali_translations)
            tts = gTTS(text=speech_text, lang="ne")
            mp3_bytes = BytesIO()
            tts.write_to_fp(mp3_bytes)
            mp3_bytes.seek(0)
        st.audio(mp3_bytes.read(), format="audio/mp3")


