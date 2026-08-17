# Medicine Label Reader

## Problem
Many customers, especially elderly ones, cannot read the English text on
medicine strips and boxes. This pipeline reads a photo of a medicine label
aloud in Nepali, so a customer can hear what the medicine is and how to
take it.

## Pipeline
This follows the same three-stage pattern as Lab 7, with OCR standing in
for image captioning:

1. **OCR (vision)** — `easyocr` extracts the English text printed on the
   label from an uploaded or camera-captured photo.
2. **Translation (NLP)** — `deep_translator` (Google Translate backend)
   converts the extracted English text into Nepali.
3. **Text-to-speech (speech synthesis)** — `gTTS` converts the Nepali text
   into a playable audio clip.

Two or more AI capabilities are combined: computer vision (text detection)
and natural language processing (translation), with speech synthesis as a
third stage — mirroring how Lab 7 chained captioning, translation, and TTS.

## Files
- `ocr_module.py` — standalone OCR test (run first, on its own)
- `translate_module.py` — standalone translation test
- `tts_module.py` — standalone TTS test
- `app.py` — combined Streamlit app
- `requirements.txt` — dependencies

## Running it
```bash
pip install -r requirements.txt
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## Observed limitations
- OCR accuracy drops on blurry photos, glare on foil packaging, or
  heavily stylized fonts used by some drug brands.
- Google Translate sometimes translates drug names literally instead of
  transliterating them, which can read oddly in Nepali.
- The app assumes label text is in English; it does not currently handle
  labels printed only in Nepali or Hindi.

## Possible extensions
- Add a rule-based check that leaves recognized drug names untranslated
  (transliterated instead), since brand names shouldn't be translated as
  ordinary words.
- Add a confidence threshold from easyocr and flag low-confidence photos
  for a retake.
