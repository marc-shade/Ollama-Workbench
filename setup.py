import subprocess
import sys

# Function to install a package using pip
def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Install required packages
required_packages = ['spacy', 'watchdog']
for package in required_packages:
    install(package)

# Install required packages from requirements.txt
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

print("Setup complete! Please ensure to activate your environment before running the application.")

# Download and install the SpaCy model
subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
