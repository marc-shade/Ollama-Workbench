"""
Text-to-Speech utilities for converting text to speech and playing audio.
"""
import os
import tempfile
from gtts import gTTS
import subprocess
import logging

# Set up logging
logging.basicConfig(
    filename='voice_utils.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Try to import pygame, but make it optional
try:
    import pygame
    pygame_available = True
    logger.info("Pygame is available for audio playback")
except ImportError:
    pygame_available = False
    logger.warning("Pygame is not available. Audio playback will be disabled.")

def zonos_text_to_speech(text, lang='en', **kwargs):
    """Convert text to speech using Zonos TTS.

    Args:
        text (str): The text to convert to speech.
        lang (str): Language code (default: 'en').
        **kwargs: Additional keyword arguments.

    Returns:
        str: Path to the generated audio file, or None if Zonos is not available or fails.
    """
    zonos_dir = '/Volumes/FILES/code/Zonos'
    if not os.path.exists(zonos_dir):
        print("Zonos TTS is not installed.")


    try:
        # Construct the command to execute the Zonos TTS script
        command = [
            'python',
            os.path.join(zonos_dir, 'main.py'),  # Assuming main.py is the main script
            '--text', text,
            '--language', lang,
            '--output_dir', tempfile.gettempdir()
        ]

        # Execute the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Extract the audio file path from the output
        audio_file = result.stdout.strip()

        return audio_file

    except subprocess.CalledProcessError as e:
        print(f"Zonos TTS failed: {e}")

    except Exception as e:
        print(f"Error using Zonos TTS: {e}")



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
        # Try Zonos TTS first
        audio_file = zonos_text_to_speech(text, lang, **kwargs)
        if audio_file:
            return audio_file

        # If Zonos is not available or fails, use gTTS as a fallback

    # Create a temporary file to store the audio
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, 'temp_speech.mp3')
        
        # Generate speech
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_file)
        
        return temp_file


    except Exception as e:
        print(f"Error generating speech: {e}")


def play_speech(audio_file):
    """
    Play the generated speech audio file.
    
    Args:
        audio_file (str): Path to the audio file to play
        
    Returns:
        bool: True if playback was successful, False otherwise
    """
    # Check if pygame is available
    if not pygame_available:
        logger.warning("Cannot play speech: pygame is not available")
        print("Audio playback is disabled because pygame is not installed.")
        print(f"Audio file saved at: {audio_file}")
        return False
        
    try:
        if not audio_file or not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return False
            
        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        logger.info(f"Successfully played audio: {audio_file}")
        return True
    except Exception as e:
        logger.error(f"Error playing speech: {e}")
        print(f"Error playing speech: {e}")
        return False
    finally:
        try:
            if pygame_available:
                pygame.mixer.quit()
        except Exception as e:
            logger.error(f"Error quitting pygame mixer: {e}")
            pass
