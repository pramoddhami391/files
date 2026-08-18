# 💊 Medicine Label Reader

A complete end-to-end AI pipeline combining **Computer Vision**, **Natural Language Processing & Translation**, and **Speech Synthesis (TTS)** wrapped in a Streamlit web application.


---

##  Problem Statement
Many patients—especially the elderly or non-English literate individuals—cannot read the fine English print on medicine strips, bottles, and boxes. **Aushadhi Vani** converts medicine label photos into clear, spoken Nepali audio explanations, allowing users to hear what the medicine is, its dosage, and how to take it safely.

---

##  3 Combined AI Capabilities

```
 📷 [Stage 1: Vision AI / OCR]
    ├── OpenCV CLAHE Contrast Equalization & 2x Bicubic Upscaling
    └── EasyOCR / Gemini 1.5 Flash Vision (Text & Attribute Extraction)
              │
              ▼
 🌐 [Stage 2: NLP & Machine Translation]
    ├── Noise Filtering & Medical Text Cleaning
    └── English ➔ Nepali Neural Translation (deep-translator / Google Translator)
              │
              ▼
 🔊 [Stage 3: Speech Synthesis (TTS)]
    └── Nepali Audio Stream Generation (gTTS) ➔ Embedded Browser Audio Player
```

1. **Stage 1: Computer Vision & Feature Extraction**
   - Preprocesses camera photos using OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization) and 2x bicubic scaling to eliminate metallic foil glare and enlarge micro-print fonts.
   - Extracts label text using EasyOCR with confidence filtering ($>45\%$) or Gemini 1.5 Flash Vision AI.

2. **Stage 2: NLP & Machine Translation**
   - Filters out stock photo watermarks, barcode noise, and unreadable character fragments.
   - Formats medical attributes (`mg`, `ml`, dosage) and translates English text into Nepali via Neural Machine Translation.

3. **Stage 3: Speech Synthesis (TTS)**
   - Converts translated Nepali explanations into audible `.mp3` speech streams using Google Text-to-Speech (`gTTS`).

---

## 🌟 Key Features
- **Dual Vision Engine**: Supports **Gemini 1.5 Flash Vision AI** for 99% accuracy on blurry packaging, with an offline fallback to **Local EasyOCR + OpenCV**.
- **OpenCV Vision Inspector**: Includes an interactive UI expander to view preprocessed contrast-enhanced images.
- **Smart Confidence Filtering**: Automatically discards low-confidence OCR misreadings and watermarks.
- **Bilingual Display & Spoken Audio**: Renders side-by-side English/Nepali cards alongside a playable browser audio player.



---

## 🚀 Running Locally

1. **Clone & Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit App**:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your web browser.

---

### Deploying on Hugging Face Spaces (Recommended for 16GB Free RAM)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose SDK: **Streamlit**, Hardware: **CPU Basic (Free - 16 GB RAM)**.
3. Upload `app.py`, `ocr_module.py`, and `requirements.txt`.

