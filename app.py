from flask import Flask, request, jsonify
from gtts import gTTS
import base64
import io
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/v1/audio/speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('input', '')
        voice = data.get('voice', 'en-US-Wavenet-A')
        
        # Extract language from voice (e.g., 'en-US-Wavenet-A' -> 'en')
        lang = voice.split('-')[0] if '-' in voice else 'en'
        
        # Create a bytes buffer for the audio
        audio_buffer = io.BytesIO()
        
        # Generate speech
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(audio_buffer)
        
        # Get the audio data and encode it
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({'audio': audio_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
