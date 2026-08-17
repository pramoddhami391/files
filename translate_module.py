"""
Part B: Translate Extracted Text to Nepali (NLP / Machine Translation)
Standalone test script.

Install:
    pip install deep-translator

Note: using deep_translator instead of googletrans (like the lab's app.py
does in Part D) because it does not require async/await and is more stable.
"""

from deep_translator import GoogleTranslator

# Step 1: Text you'd normally get from ocr_module.py
english_text = "Paracetamol 500mg Tablet. Take one tablet after meals."

# Step 2: Translate English to Nepali
translated_text = GoogleTranslator(source='en', target='ne').translate(english_text)

print("English:", english_text)
print("Nepali:", translated_text)
