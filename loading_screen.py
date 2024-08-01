import time
import os
import sys
import random
import threading
import subprocess
import cursor

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_centered(text):
    terminal_width = os.get_terminal_size().columns
    print(text.center(terminal_width))

def loading_animation():
    frames = [
        r"""
                                  .@#.                   
            +=..                  -@@%..                 
          .*@@.                  *+@%#= .                
           *@#*+               .@:%@@** :                
          .*@@@.=%.     .     .%.*@@@**.:                
           =@@@@: -* .+#@-#%@=*-.@@@%*+.:                
            *@@@@=.+#=:.     ++:*@@@**-..                
            :%@@@@+.        .++-+@@%++.:                 
             .#@@@+..      ::  :=@%+*.:.                 
             . .@%#@+-+*=#+.     .-%#..                  
              -%*=*%@@#=:          :#-.                  
            .#@#*=---:   .+*#%%%#-  .*:                  
             .=@@@*:    :=#@@@#@@+.  +* .                
             .:=@@*      :%%%@@@#    .*..                
                #@.        -*##+      =* .               
                =.                 .  :@:.               
               =@:-*#:            :    #= .              
               =+@#.    ::      +-     =@:.              
               *:+-   +--:::-+#+       .%+ .             
                :@@@@+-@@@@@#+-         -% .             
                  *@@#. @@@#+:   .      -@= .            
                  :--:. =@#*+=.   .    :   :.            
                        -@#++=::     .  :-.              
                         @@*++==-    .-:                 
                         :#*=-:.  :-.                    
                         .::::::.                        
         """,
        r"""
             .*##-                     -###.             
             *+-:*                     *=.-*             
            :+#@-.*:                 :*=-@*=:            
            -+@@@* -+.              +=:*@@%--            
            :+%@@@+ .+.  *%#*%@-  .*-:+@@@#=:            
             **@@@@=  ++@@%%##*#*-#-:+@@@@+*             
             :*#@@@@-.--#@@%###%- --+@@@@#*:             
              :#+@@@%-.:-+%@@@@+.  .#@@@+#:              
                =*+*-:....::::..    .++*=                
                  *=:.               :=                  
                 :**##+:....:...::=##++:                 
                 =@@@@%#=--:   :-%@#%@%-                 
                  .*%@@@#-::    *@@@#+.                  
                  .+*@%+:::.     =%%==.                  
                  -=---:*+#*-+*-: ...:-                  
                  :*-=--::-*#=.   ...=:                  
                   =+--:*-:-=. .-   -=                   
                    =#=--=+%%*:..:-++                    
                    =+=*#%**#*++=-.:-                    
                   .**-=***#*+:.   =*                    
                    -+--====-..  . :-                    
                   .**====-:.....  =*                    
                    =+===-::....   :=                    
                    :--=----....  . .                    
                         .: .                            

         """,
    ]
    loading_text = [
        "Booting Up...",
        "Initializing Flux Capacitor...",
        "Loading AI Modules...",
        "Warming Up GPUs...",
        "Calibrating Quantum Sensors...",
        "Generating Creative Sparks...",
        "Connecting to the Matrix...",
        "Downloading Latest Stuff...",
        "First time?",
        "Oh boy, this might take a while...",
        "Getting the latest Ollama models...",
        "Preparing for Awesomeness..."
    ]
    dots = 0
    max_dots = 12

    while not installation_complete.is_set():
        clear_screen()
        print("\n" * 5)
        print_centered(random.choice(frames))
        print("\n" * 2)
        print_centered(f"{random.choice(loading_text)}{'.' * dots}")
        print("\n" * 5)
        time.sleep(0.5)
        dots = (dots + 1) % (max_dots + 1)

def run_installation(command):
    # Hide the cursor during installation
    cursor.hide()

    # Run the command, capturing output to suppress it
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    installation_complete.set()

    # Show the cursor again after installation
    cursor.show()

def run_installation_with_output(command):
    # Run the command, showing output
    subprocess.run(command, shell=True)
    installation_complete.set()

def print_completion_message():
    clear_screen()
    print("\n" * 5)
    print("Installation/Update Complete!")
    print("______________________________________________________")

    print("Built by 🟦🟦 2 Acre Studios")
    print("With thanks to:")
    print("Ollama")
    print("SerpApi")
    print("Google")
    print("Streamlit")
    print("LangChain")
    print("AutoGen")
    print("Chroma")
    print("Hugging Face")
    print("PyTorch")
    print("DuckDuckGo")
    print("Bing")
    print("And the countless developers and researchers")
    print("whose work made this possible.")
    print("______________________________________________________")

    print("Launching Ollama Workbench...")
    print("\n" * 5)
    time.sleep(2)

if __name__ == "__main__":
    installation_complete = threading.Event()
    
    if len(sys.argv) > 1:
        installation_command = " ".join(sys.argv[1:])
        
        if "--no-loading-screen" in sys.argv:
            # Run installation with output
            run_installation_with_output(installation_command)
        else:
            # Run installation with loading screen
            installation_thread = threading.Thread(target=run_installation, args=(installation_command,))
            installation_thread.start()
            loading_animation()
            installation_thread.join()
        
        print_completion_message()
    else:
        print("Usage: python loading_screen.py <installation_command>")
        sys.exit(1)