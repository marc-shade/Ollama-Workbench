import time
import os
import sys
import random
import threading

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_centered(text):
    terminal_width = os.get_terminal_size().columns
    print(text.center(terminal_width))

def dancing_llama():
    frames = [
        r"""
         /\____/\
        (  oo   )
        (  ==   )
         \____/   ❤️
        """,
        r"""
         /\____/\
        (  oo   )
        (  ==   )
         \____/ ❤️
        """,
        r"""
         /\____/\
        (  oo   )
        (  ==   )
       ❤️ \____/
        """,
        r"""
         /\____/\
        (  oo   )
        (  ==   )
        ❤️\____/
        """
    ]
    return frames

def loading_animation():
    llama_frames = dancing_llama()
    loading_text = "Installing Ollama Workbench"
    dots = 0
    max_dots = 3

    while not installation_complete.is_set():
        clear_screen()
        print("\n" * 5)
        print_centered(llama_frames[dots % len(llama_frames)])
        print("\n" * 2)
        print_centered(f"{loading_text}{'.' * dots}")
        print("\n" * 5)
        time.sleep(0.5)
        dots = (dots + 1) % (max_dots + 1)

def run_installation(command):
    os.system(command)
    installation_complete.set()

if __name__ == "__main__":
    installation_complete = threading.Event()
    
    if len(sys.argv) > 1:
        installation_command = " ".join(sys.argv[1:])
        installation_thread = threading.Thread(target=run_installation, args=(installation_command,))
        installation_thread.start()
        
        loading_animation()
        
        installation_thread.join()
    else:
        print("Usage: python loading_screen.py <installation_command>")
        sys.exit(1)

    clear_screen()
    print("\n" * 5)
    print_centered("Installation Complete!")
    print_centered("Launching Ollama Workbench...")
    print("\n" * 5)
    time.sleep(2)