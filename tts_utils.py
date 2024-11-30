"""
Text-to-Speech utilities for converting text to speech and playing audio.
"""
import os
import tempfile
from gtts import gTTS
import pygame

def text_to_speech(text, lang='en', **kwargs):
    """
    Convert text to speech using Google Text-to-Speech.
    
    Args:
        text (str): The text to convert to speech
        lang (str): Language code (default: 'en' for English)
        **kwargs: Additional arguments that will be ignored
        
    Returns:
        str: Path to the generated audio file
    """
    try:
        # Create a temporary file to store the audio
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, 'temp_speech.mp3')
        
        # Generate speech
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_file)
        
        return temp_file
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def play_speech(audio_file):
    """
    Play the generated speech audio file.
    
    Args:
        audio_file (str): Path to the audio file to play
        
    Returns:
        bool: True if playback was successful, False otherwise
    """
    try:
        if not audio_file or not os.path.exists(audio_file):
            return False
            
        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        return True
    except Exception as e:
        print(f"Error playing speech: {e}")
        return False
    finally:
        try:
            pygame.mixer.quit()
        except:
            pass
