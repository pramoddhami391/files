"""
Part B: Translate Extracted Text to Nepali (NLP / Machine Translation)
Standalone test script.

Install:
    pip install deep-translator
"""

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound

translator = GoogleTranslator(source='en', target='ne')

def translate_to_nepali(text):
    try:
        return translator.translate(text)
    except TranslationNotFound:
        return text
    except Exception:
        return text

# Step 1: Text you'd normally get from ocr_module.py
english_text = "Paracetamol 500mg Tablet. Take one tablet after meals."

# Step 2: Translate English to Nepali safely
translated_text = translate_to_nepali(english_text)

print("English:", english_text)
print("Nepali:", translated_text)

