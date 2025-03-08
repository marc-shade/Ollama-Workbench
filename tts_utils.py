"""
Text-to-Speech utilities for converting text to speech and playing audio.
"""
import os
import tempfile
try:
    from gtts import gTTS
except ImportError:
    print("Warning: gtts package not found, using fallback implementation")
    # Fallback implementation for gtts
    class gTTS:
        def __init__(self, text="", lang="en", **kwargs):
            self.text = text
            self.lang = lang
            print(f"Warning: Using fallback gTTS with text: {text[:50]}...")
            
        def save(self, filename):
            print(f"Warning: Saving fallback speech to {filename}")
            # Create an empty file
            with open(filename, 'w') as f:
                f.write("# This is a placeholder file created by the fallback gTTS implementation")
            return filename
try:
    import pygame
except ImportError:
    print("Warning: pygame package not found, using fallback implementation")
    # Fallback implementation for pygame
    class pygame:
        class mixer:
            @staticmethod
            def init():
                print("Warning: Using fallback pygame.mixer.init()")
                
            @staticmethod
            def quit():
                print("Warning: Using fallback pygame.mixer.quit()")
                
            class music:
                @staticmethod
                def load(filename):
                    print(f"Warning: Using fallback pygame.mixer.music.load({filename})")
                    
                @staticmethod
                def play():
                    print("Warning: Using fallback pygame.mixer.music.play()")
                    
                @staticmethod
                def get_busy():
                    # Return False to exit the playback loop immediately
                    return False
                    
        class time:
            class Clock:
                def tick(self, framerate):
                    pass

import subprocess

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
        except Exception as e:
            print(f"Error quitting pygame mixer: {e}")
            pass
