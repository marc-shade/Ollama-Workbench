"""
Enhanced voice input/output utilities for Ollama Workbench.
"""
import os
import tempfile
import time
import threading
import queue
import sys
import json
import wave
import pyaudio
import speech_recognition as sr
from gtts import gTTS
import pygame
import numpy as np
import requests
import logging
import subprocess

logger = logging.getLogger('voice_utils')

# Voice configuration
DEFAULT_VOICE = {
    'provider': 'gtts',  # gtts, elevenlabs, tts_server
    'voice_id': 'en-US-Wavenet-D',
    'language': 'en',
    'speed': 1.0,
    'pitch': 1.0
}

# Constants
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500  # Silence threshold for voice activity detection
MAX_SILENCE_DURATION = 2.0  # Max silence duration in seconds to stop recording

# Queue for real-time processing
audio_queue = queue.Queue()
is_recording = False
stop_recording = threading.Event()

class VoiceManager:
    """Manages voice input and output for the application."""
    
    def __init__(self):
        """Initialize the voice manager."""
        self.audio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.stream = None
        self.frames = []
        self.silence_start_time = None
        self.is_listening = False
        self.listening_thread = None
        self.processing_thread = None
        self.speech_callback = None
        self.error_callback = None
        
        # Load voice settings
        self.voices = self._load_voice_settings()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
    def _load_voice_settings(self):
        """Load voice settings from file."""
        voice_settings_file = "voice_settings.json"
        voices = {}
        
        if os.path.exists(voice_settings_file):
            try:
                with open(voice_settings_file, 'r') as f:
                    voices = json.load(f)
            except Exception as e:
                logger.error(f"Error loading voice settings: {e}")
                voices = {
                    'default': DEFAULT_VOICE
                }
        else:
            # Create default settings
            voices = {
                'default': DEFAULT_VOICE,
                'male': {
                    'provider': 'gtts',
                    'voice_id': 'en-US-Wavenet-D',
                    'language': 'en',
                    'speed': 1.0,
                    'pitch': 1.0
                },
                'female': {
                    'provider': 'gtts',
                    'voice_id': 'en-US-Wavenet-A',
                    'language': 'en',
                    'speed': 1.0,
                    'pitch': 1.0
                }
            }
            
            # Save default settings
            try:
                with open(voice_settings_file, 'w') as f:
                    json.dump(voices, f, indent=4)
            except Exception as e:
                logger.error(f"Error saving default voice settings: {e}")
                
        return voices
    
    def save_voice_settings(self):
        """Save voice settings to file."""
        voice_settings_file = "voice_settings.json"
        try:
            with open(voice_settings_file, 'w') as f:
                json.dump(self.voices, f, indent=4)
            logger.info("Voice settings saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving voice settings: {e}")
            return False
    
    def get_available_voices(self):
        """Get a list of available voice profiles."""
        return list(self.voices.keys())
    
    def add_voice_profile(self, name, settings):
        """Add a new voice profile."""
        if name in self.voices:
            logger.warning(f"Voice profile '{name}' already exists and will be overwritten")
        
        self.voices[name] = settings
        self.save_voice_settings()
        return True
    
    def remove_voice_profile(self, name):
        """Remove a voice profile."""
        if name == 'default':
            logger.error("Cannot remove default voice profile")
            return False
        
        if name in self.voices:
            del self.voices[name]
            self.save_voice_settings()
            return True
        else:
            logger.warning(f"Voice profile '{name}' does not exist")
            return False
    
    def get_voice_settings(self, voice_name='default'):
        """Get settings for a specific voice."""
        return self.voices.get(voice_name, self.voices['default'])
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio input stream."""
        audio_queue.put(in_data)
        self.frames.append(in_data)
        
        # Check for silence
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
            if self.silence_start_time is None:
                self.silence_start_time = time.time()
            elif time.time() - self.silence_start_time > MAX_SILENCE_DURATION:
                stop_recording.set()
        else:
            self.silence_start_time = None
            
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_queue(self):
        """Process audio data from the queue in real-time."""
        while not stop_recording.is_set() or not audio_queue.empty():
            try:
                data = audio_queue.get(timeout=1.0)
                # Here you could implement real-time processing
                # For example, sending chunks to a streaming ASR service
                audio_queue.task_done()
            except queue.Empty:
                continue
    
    def start_listening(self, speech_callback=None, error_callback=None):
        """Start listening for voice input."""
        if self.is_listening:
            logger.warning("Already listening")
            return False
        
        self.speech_callback = speech_callback
        self.error_callback = error_callback
        self.frames = []
        self.silence_start_time = None
        stop_recording.clear()
        
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self.is_listening = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_queue)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info("Voice input started")
            return True
        except Exception as e:
            logger.error(f"Error starting voice input: {e}")
            if error_callback:
                error_callback(str(e))
            return False
    
    def stop_listening(self):
        """Stop listening and process the recorded audio."""
        if not self.is_listening:
            logger.warning("Not currently listening")
            return False
        
        try:
            stop_recording.set()
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            self.is_listening = False
            
            # Process the recorded audio
            if self.frames:
                # Save the recorded audio to a temporary WAV file
                temp_file = os.path.join(tempfile.gettempdir(), 'recorded_audio.wav')
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(self.frames))
                
                # Recognize speech
                try:
                    with sr.AudioFile(temp_file) as source:
                        audio_data = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio_data)
                        
                        if self.speech_callback:
                            self.speech_callback(text)
                        
                        logger.info(f"Recognized speech: {text}")
                        return text
                except sr.UnknownValueError:
                    error_msg = "Could not understand audio"
                    logger.warning(error_msg)
                    if self.error_callback:
                        self.error_callback(error_msg)
                except sr.RequestError as e:
                    error_msg = f"Speech recognition service error: {e}"
                    logger.error(error_msg)
                    if self.error_callback:
                        self.error_callback(error_msg)
            
            return ""
        except Exception as e:
            logger.error(f"Error stopping voice input: {e}")
            if self.error_callback:
                self.error_callback(str(e))
            return False
    
    def text_to_speech(self, text, voice_name='default', output_file=None):
        """
        Convert text to speech using the specified voice profile.
        
        Args:
            text (str): The text to convert to speech.
            voice_name (str): The name of the voice profile to use.
            output_file (str): Optional path to save the output file.
            
        Returns:
            str: Path to the generated audio file.
        """
        if not text:
            logger.warning("No text provided for TTS")
            return None
        
        voice_settings = self.get_voice_settings(voice_name)
        provider = voice_settings.get('provider', 'gtts')
        
        if not output_file:
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, f'speech_{int(time.time())}.mp3')
        
        try:
            if provider == 'gtts':
                return self._gtts_text_to_speech(text, voice_settings, output_file)
            elif provider == 'elevenlabs':
                return self._elevenlabs_text_to_speech(text, voice_settings, output_file)
            elif provider == 'tts_server':
                return self._tts_server_text_to_speech(text, voice_settings, output_file)
            else:
                logger.warning(f"Unknown TTS provider: {provider}, falling back to gTTS")
                return self._gtts_text_to_speech(text, voice_settings, output_file)
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")
            # Fallback to gTTS if other methods fail
            try:
                return self._gtts_text_to_speech(text, voice_settings, output_file)
            except Exception as e2:
                logger.error(f"Error in fallback gTTS: {e2}")
                return None
    
    def _gtts_text_to_speech(self, text, voice_settings, output_file):
        """Convert text to speech using Google Text-to-Speech."""
        lang = voice_settings.get('language', 'en')
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)
        return output_file
    
    def _elevenlabs_text_to_speech(self, text, voice_settings, output_file):
        """Convert text to speech using ElevenLabs API."""
        api_key = os.environ.get('ELEVENLABS_API_KEY')
        if not api_key:
            logger.warning("ElevenLabs API key not found, falling back to gTTS")
            return self._gtts_text_to_speech(text, voice_settings, output_file)
        
        voice_id = voice_settings.get('voice_id', '21m00Tcm4TlvDq8ikWAM')  # Default ElevenLabs voice ID
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return output_file
        else:
            logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return self._gtts_text_to_speech(text, voice_settings, output_file)
    
    def _tts_server_text_to_speech(self, text, voice_settings, output_file):
        """Convert text to speech using local TTS server."""
        voice_name = voice_settings.get('voice_id', 'default')
        
        # Check if server is running by pinging healthcheck endpoint
        try:
            healthcheck_response = requests.get("http://localhost:8000/healthcheck", timeout=2)
            if healthcheck_response.status_code != 200:
                logger.warning("TTS server healthcheck failed, starting server...")
                self._start_tts_server()
                time.sleep(2)  # Wait for server to start
        except requests.exceptions.RequestException as e:
            logger.warning(f"TTS server not responding ({e}), starting server...")
            self._start_tts_server()
            time.sleep(2)  # Wait for server to start
        
        # Try to synthesize speech
        url = "http://localhost:8000/synthesize"
        data = {
            "text": text,
            "voice": voice_name,
            "speed": voice_settings.get('speed', 1.0),
            "pitch": voice_settings.get('pitch', 1.0)
        }
        
        try:
            # First, try to reload profiles to ensure server has the latest
            try:
                reload_response = requests.post("http://localhost:8000/reload_profiles", timeout=2)
                if reload_response.status_code == 200:
                    logger.debug("Voice profiles reloaded on TTS server")
            except Exception:
                logger.debug("Failed to reload voice profiles on TTS server")
            
            # Now request speech synthesis
            response = requests.post(url, json=data, timeout=5)
            
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                return output_file
            else:
                logger.error(f"TTS server error: {response.status_code} - {response.text}")
                return self._gtts_text_to_speech(text, voice_settings, output_file)
        except Exception as e:
            logger.error(f"TTS server connection error: {e}")
            return self._gtts_text_to_speech(text, voice_settings, output_file)
    
    def _start_tts_server(self):
        """Start the TTS server if it's not running."""
        try:
            # Get path to TTS server directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tts_server_dir = os.path.join(script_dir, "tts_server")
            start_script = os.path.join(tts_server_dir, "start_tts_server.sh")
            
            # Make sure start script is executable
            os.chmod(start_script, 0o755)
            
            # Start the server
            subprocess.Popen(['bash', start_script],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
            
            logger.info("TTS server started")
            return True
        except Exception as e:
            logger.error(f"Error starting TTS server: {e}")
            return False
    
    def play_speech(self, audio_file, block=True):
        """
        Play the generated speech audio file.
        
        Args:
            audio_file (str): Path to the audio file to play.
            block (bool): Whether to block until playback is complete.
            
        Returns:
            bool: True if playback was successful, False otherwise.
        """
        try:
            if not audio_file or not os.path.exists(audio_file):
                logger.warning(f"Audio file does not exist: {audio_file}")
                return False
            
            # Initialize pygame mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            if block:
                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            
            return True
        except Exception as e:
            logger.error(f"Error playing speech: {e}")
            return False
    
    def stop_playback(self):
        """Stop any currently playing audio."""
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            return True
        except Exception as e:
            logger.error(f"Error stopping playback: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if self.audio:
                self.audio.terminate()
            
            if pygame.mixer.get_init():
                pygame.mixer.quit()
                
            logger.info("Voice manager resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Singleton instance
voice_manager = VoiceManager()

# Convenience functions that use the singleton
def start_voice_input(callback=None, error_callback=None):
    """Start voice input and return the recognized text."""
    return voice_manager.start_listening(callback, error_callback)

def stop_voice_input():
    """Stop voice input and process the recorded audio."""
    return voice_manager.stop_listening()

def text_to_speech(text, voice_name='default', output_file=None):
    """Convert text to speech using the specified voice."""
    return voice_manager.text_to_speech(text, voice_name, output_file)

def play_speech(audio_file, block=True):
    """Play the audio file."""
    return voice_manager.play_speech(audio_file, block)

def stop_speech():
    """Stop any currently playing speech."""
    return voice_manager.stop_playback()

def get_available_voices():
    """Get list of available voice profiles."""
    return voice_manager.get_available_voices()

def get_voice_settings(voice_name='default'):
    """Get settings for a specific voice."""
    return voice_manager.get_voice_settings(voice_name)

def add_voice_profile(name, settings):
    """Add a new voice profile."""
    return voice_manager.add_voice_profile(name, settings)

def remove_voice_profile(name):
    """Remove a voice profile."""
    return voice_manager.remove_voice_profile(name)

# Cleanup function to be called at program exit
def cleanup():
    """Clean up voice manager resources."""
    voice_manager.cleanup()

# Register cleanup handler
import atexit
atexit.register(cleanup)