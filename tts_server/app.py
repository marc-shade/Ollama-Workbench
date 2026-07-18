from flask import Flask, request, jsonify, send_file
from werkzeug.exceptions import BadRequest
from gtts import gTTS
from gtts.lang import tts_langs
import base64
import io
import os
import time
import logging
import uuid
import json
from flask_cors import CORS
import tempfile
from threading import Lock

# Resolve paths relative to this file so the server works regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'tts_server.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('tts_server')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create cache directory
CACHE_DIR = os.path.join(tempfile.gettempdir(), "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache lock for thread safety
cache_lock = Lock()

# Voice configurations
VOICE_PROFILES = {
    "default": {
        "provider": "gtts",
        "language": "en",
        "voice_id": "en-US-Wavenet-D",
        "speed": 1.0,
        "pitch": 1.0
    }
}

# Load voice profiles
def load_voice_profiles():
    global VOICE_PROFILES
    voice_profiles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voice_settings.json")
    if os.path.exists(voice_profiles_path):
        try:
            with open(voice_profiles_path, 'r') as f:
                VOICE_PROFILES = json.load(f)
            logger.info(f"Loaded {len(VOICE_PROFILES)} voice profiles")
        except Exception as e:
            logger.error(f"Error loading voice profiles: {e}")

# Try to load existing profiles
try:
    load_voice_profiles()
except Exception as e:
    logger.error(f"Error during initialization: {e}")

# Cache management
def get_cached_audio(text, voice_profile):
    """Get cached audio file if it exists"""
    with cache_lock:
        cache_key = f"{text}_{json.dumps(voice_profile)}"
        import hashlib
        hashed_key = hashlib.md5(cache_key.encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{hashed_key}.mp3")
        
        if os.path.exists(cache_path):
            # Check if cache is recent (less than 24 hours)
            if time.time() - os.path.getmtime(cache_path) < 86400:
                return cache_path
    return None

def save_to_cache(audio_data, text, voice_profile):
    """Save audio data to cache"""
    with cache_lock:
        cache_key = f"{text}_{json.dumps(voice_profile)}"
        import hashlib
        hashed_key = hashlib.md5(cache_key.encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{hashed_key}.mp3")
        
        with open(cache_path, 'wb') as f:
            f.write(audio_data)
        return cache_path

# Clean the cache periodically
def clean_cache():
    """Remove cache files older than 7 days"""
    now = time.time()
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > 604800:  # 7 days
            os.remove(file_path)

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'timestamp': time.time()})

@app.route('/voices', methods=['GET'])
def get_voices():
    """Get available voices"""
    try:
        # Get supported languages from gTTS
        languages = tts_langs()
        
        # Get custom voice profiles
        profiles = list(VOICE_PROFILES.keys())
        
        return jsonify({
            'supported_languages': languages,
            'voice_profiles': profiles
        })
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Main endpoint for text-to-speech conversion"""
    try:
        data = request.json
        text = data.get('text', '')
        voice_name = data.get('voice', 'default')
        speed = data.get('speed', 1.0)
        pitch = data.get('pitch', 1.0)
        
        # Get voice settings
        voice_settings = VOICE_PROFILES.get(voice_name, VOICE_PROFILES['default'])
        
        # Override settings if provided
        voice_settings['speed'] = speed
        voice_settings['pitch'] = pitch
        
        # Try to get from cache first
        cached_file = get_cached_audio(text, voice_settings)
        if cached_file:
            logger.info(f"Serving cached audio for: {text[:30]}...")
            return send_file(cached_file, mimetype='audio/mpeg')
        
        # Create a bytes buffer for the audio
        audio_buffer = io.BytesIO()
        
        # Generate speech based on provider
        provider = voice_settings.get('provider', 'gtts')
        if provider == 'gtts':
            lang = voice_settings.get('language', 'en')
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.write_to_fp(audio_buffer)
        else:
            # Default to gTTS if provider not supported
            lang = voice_settings.get('language', 'en')
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.write_to_fp(audio_buffer)
        
        # Get the audio data
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        
        # Save to cache
        cache_path = save_to_cache(audio_data, text, voice_settings)
        
        # Return the audio file
        return send_file(
            cache_path,
            mimetype='audio/mpeg'
        )
    except BadRequest:
        # Malformed request payload is a client error, not a server error
        raise
    except Exception as e:
        logger.error(f"Error in synthesize: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/audio/speech', methods=['POST'])
def text_to_speech():
    """OpenAI-compatible endpoint for text-to-speech conversion"""
    try:
        data = request.json
        text = data.get('input', '')
        voice = data.get('voice', 'en-US-Wavenet-A')
        
        # Extract language from voice (e.g., 'en-US-Wavenet-A' -> 'en');
        # a bare language code like 'fr' is used as-is
        lang = voice.split('-')[0] if '-' in voice else (voice or 'en')
        
        # Create a bytes buffer for the audio
        audio_buffer = io.BytesIO()
        
        # Generate speech
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(audio_buffer)
        
        # Get the audio data and encode it
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({'audio': audio_base64})
    except BadRequest:
        # Malformed request payload is a client error, not a server error
        raise
    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reload_profiles', methods=['POST'])
def reload_profiles():
    """Reload voice profiles from disk"""
    try:
        load_voice_profiles()
        return jsonify({'status': 'success', 'message': 'Voice profiles reloaded', 'count': len(VOICE_PROFILES)})
    except Exception as e:
        logger.error(f"Error reloading profiles: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clean_cache', methods=['POST'])
def trigger_clean_cache():
    """Manually trigger cache cleaning"""
    try:
        clean_cache()
        return jsonify({'status': 'success', 'message': 'Cache cleaned'})
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cache_stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    try:
        files = os.listdir(CACHE_DIR)
        total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files if os.path.isfile(os.path.join(CACHE_DIR, f)))
        
        return jsonify({
            'file_count': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        })
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({'error': str(e)}), 500

# Clean cache on startup
try:
    clean_cache()
except Exception as e:
    logger.error(f"Error cleaning cache on startup: {e}")

if __name__ == '__main__':
    # Run the Flask app. Port 5002 - port 8000 belongs to the OpenAI
    # compatibility server started by the main Streamlit app; the two
    # collided when both used 8000.
    app.run(host='127.0.0.1', port=int(os.environ.get('TTS_SERVER_PORT', 5002)))
