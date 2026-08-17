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

# Cache the OCR reader so it only loads once per session, same idea as
# @st.cache_resource on the BLIP model in the lab's app.py
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

    # --- Stage 1: OCR (vision) ---
    st.subheader("Extracted text")
    with st.spinner("Reading label..."):
        image_array = np.array(image)
        results = reader.readtext(image_array, detail=0)
        english_text = " ".join(results)

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
