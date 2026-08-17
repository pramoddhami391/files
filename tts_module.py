"""
Part C: Convert Nepali Text to Speech (Speech Synthesis / TTS)
Standalone test script. Same pattern as the lab's tts.py.

Install:
    pip install gtts
"""

from gtts import gTTS

# Step 1: Nepali text you'd normally get from translate_module.py
nepali_text = "प्यारासिटामोल ५०० मिलीग्राम ट्याब्लेट। खानापछि एक ट्याब्लेट लिनुहोस्।"

# Step 2: Convert text to speech
tts = gTTS(text=nepali_text, lang='ne')

# Step 3: Save as mp3
tts.save("nepali_speech.mp3")

print("Saved nepali_speech.mp3")
