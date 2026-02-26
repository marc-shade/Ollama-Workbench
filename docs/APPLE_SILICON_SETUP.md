# Ollama Workbench for Apple Silicon

This guide explains the optimized setup for Ollama Workbench on Apple Silicon (M1/M2/M3/M4) Macs.

## Improvements Made

1. **Apple Silicon Optimizations**
   - PyTorch 2.2.0 with Metal Performance Shaders (MPS) support for GPU acceleration
   - TorchVision 0.17.0 compatible with the MPS-enabled PyTorch
   - Fixed compatibility issues with sentence-transformers
   - NumPy 1.x compatibility to avoid crashes

2. **Dependency Management**
   - Switched to Poetry for cleaner dependency management
   - Eliminated conda/poetry mix that caused conflicts
   - Pinned specific versions known to work together

3. **Improved Installation Process**
   - Clear, color-coded terminal output
   - Better error messages and recovery steps
   - Automatic hardware detection
   - Automatic installation of system dependencies (wkhtmltopdf)
   - Validation tests to confirm proper setup

4. **Bug Fixes**
   - Fixed the `operator torchvision::nms does not exist` error
   - Created placeholder for missing `groq_utils.py`
   - Handled dependency chain issues with torch/transformers

## Installation Guide

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or higher

### Installation Steps

1. **Run the setup script**

   ```bash
   bash setup.sh
   ```

   This will:
   - Check if you're running on Apple Silicon
   - Install Poetry if it's not already installed
   - Configure the Poetry environment
   - Install all dependencies with the correct versions
   - Install wkhtmltopdf if needed (for PDF generation)
   - Run validation tests

2. **Verify the installation**

   ```bash
   python validate_installation.py
   ```

   This will check:
   - PyTorch with MPS support
   - All required dependencies
   - Ollama server availability

3. **Run Ollama Workbench**

   ```bash
   bash run_ollama_workbench.sh
   ```

## Troubleshooting

### NumPy Compatibility Issues

If you see errors related to NumPy compatibility:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.3 as it may crash.
```

This is because some packages are not yet compatible with NumPy 2.x. To fix this:

1. Downgrade NumPy to version 1.x:
   ```bash
   poetry run pip install "numpy>=1.24.3,<2.0.0"
   ```

2. Reinstall any packages that were giving errors:
   ```bash
   poetry run pip install -U spacy==3.7.2
   ```

### MPS Acceleration Not Working

If the validation test shows MPS is not available:

1. Make sure you're using Python 3.11 or higher
2. Try reinstalling PyTorch with:
   ```bash
   poetry run pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
   ```

### Missing Dependencies

If you're seeing import errors:

1. Run the setup script again
2. Check the Poetry environment is activated correctly
3. Manually install the specific package:
   ```bash
   poetry run pip install <package-name>==<version>
   ```

### Matplotlib Issues

If you encounter errors related to matplotlib:

1. Make sure matplotlib is installed:
   ```bash
   poetry run pip install matplotlib==3.9.2
   ```
2. If you see backend errors, try setting a non-interactive backend:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Use non-interactive backend
   import matplotlib.pyplot as plt
   ```

### Selenium Issues

If you encounter errors related to selenium:

1. Make sure selenium is installed:
   ```bash
   poetry run pip install selenium==4.24.0
   ```
2. You may need to install a webdriver. The setup script installs webdriver-manager which can handle this automatically:
   ```python
   from selenium import webdriver
   from selenium.webdriver.chrome.service import Service
   from webdriver_manager.chrome import ChromeDriverManager
   
   # This will automatically download and use the appropriate driver
   driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
   ```

### PDF Processing Issues

If you encounter errors related to PDF processing:

1. Make sure PyPDF2 is installed:
   ```bash
   poetry run pip install PyPDF2==3.0.1
   ```
2. For PDF generation, ensure both pdfkit and wkhtmltopdf are installed:
   ```bash
   poetry run pip install pdfkit==1.0.0
   ```
3. If you see import errors with PyPDF2, check the import statement:
   ```python
   # For PyPDF2 version 3.0.1
   from PyPDF2 import PdfMerger, PdfReader, PdfWriter
   
   # For older versions of PyPDF2
   from PyPDF2 import PdfFileMerger, PdfFileReader, PdfFileWriter
   ```
4. Make sure fpdf is installed for PDF generation:
   ```bash
   poetry run pip install fpdf==1.7.2
   ```
5. If you see import errors with fpdf, check the import statement:
   ```python
   # Correct import for fpdf
   from fpdf import FPDF
   
   # Example usage
   pdf = FPDF()
   pdf.add_page()
   pdf.set_font("Arial", size=12)
   pdf.cell(200, 10, txt="Hello World", ln=1, align="C")
   pdf.output("simple_demo.pdf")
   ```

### PDF Generation Issues (wkhtmltopdf)

If you're having issues with PDF generation or see errors related to wkhtmltopdf:

1. The setup script should automatically install wkhtmltopdf on macOS
2. If installation failed, you can manually install it:
   ```bash
   # Download the package
   curl -L https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-2/wkhtmltox-0.12.6-2.macos-cocoa.pkg -o wkhtmltopdf.pkg
   
   # Install it
   sudo installer -pkg wkhtmltopdf.pkg -target /
   ```
3. Verify the installation:
   ```bash
   which wkhtmltopdf
   wkhtmltopdf --version
   ```

### Code Analysis Issues

If you encounter errors related to code analysis tools like radon or flake8:

1. Make sure radon is installed:
   ```bash
   poetry run pip install radon==6.0.1
   ```
2. If you see import errors with radon, check the import statement:
   ```python
   # Correct imports for radon
   from radon.complexity import cc_visit, cc_rank
   from radon.metrics import mi_visit
   ```
3. For command-line usage:
   ```bash
   # Analyze code complexity
   poetry run radon cc path/to/file.py
   
   # Analyze maintainability index
   poetry run radon mi path/to/file.py
   ```
4. Make sure flake8 is installed for code linting:
   ```bash
   poetry run pip install flake8==7.1.1
   ```
5. If you see import errors with flake8, check the import statement:
   ```python
   # Correct import for flake8 API
   from flake8.api import legacy as flake8
   
   # Example usage
   style_guide = flake8.get_style_guide()
   report = style_guide.check_files(['path/to/file.py'])
   ```
6. For command-line usage:
   ```bash
   # Lint a file or directory
   poetry run flake8 path/to/file.py
   ```

### Autogen/Pyautogen Issues

If you encounter errors related to autogen or ag2:

1. Make sure both packages are installed:
   ```bash
   poetry run pip install autogen==0.2.35 ag2==0.2.35
   ```
2. The codebase uses autogen, so make sure to use the correct import statements:
   ```python
   # Correct import for autogen
   from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
   from autogen.agentchat.contrib.capabilities.teachability import Teachability
   ```
3. If you encounter import errors with autogen, you might need to check if the package is installed correctly:
   ```bash
   # Check if autogen is installed
   poetry run pip list | grep autogen
   ```
4. Note that autogen and ag2 are different packages with similar functionality. The project uses autogen.

### Streamlit Extensions Issues

If you encounter errors related to streamlit-extras:

1. Make sure streamlit-extras is installed:
   ```bash
   poetry run pip install streamlit-extras==0.3.6
   ```
2. If you see import errors with specific components, check if they're available in the installed version:
   ```python
   # Some components might not be available in all versions
   # For example, bottom_container is not available in 0.3.6
   from streamlit_extras.colored_header import colored_header
   from streamlit_extras.switch_page_button import switch_page
   ```
3. For missing components, consider using standard Streamlit alternatives:
   ```python
   # Instead of bottom_container
   prompt = st.chat_input("Enter your message")
   
   # Instead of colored_header
   st.markdown("<h2 style='color: blue;'>Colored Header</h2>", unsafe_allow_html=True)
   ```
4. Some components may require additional dependencies, install them as needed.

### Dependency Conflicts

If you see errors about dependency conflicts (especially with tiktoken and langchain):

1. Update tiktoken to a compatible version:
   ```bash
   poetry run pip install "tiktoken>=0.7.0,<1.0.0"
   ```

2. Install dependencies in the correct order:
   ```bash
   poetry run pip install langchain-core==0.2.15
   poetry run pip install langchain==0.2.15
   poetry run pip install langchain-community==0.2.15
   poetry run pip install langchain-openai==0.1.1
   ```

### Ollama Server Issues

If Ollama server checks fail:

1. Make sure Ollama is installed:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Start the Ollama server manually:
   ```bash
   ollama serve
   ```

## Technical Details

### Key Dependency Versions

- torch: 2.2.0 (with MPS support)
- torchvision: 0.17.0
- transformers: 4.38.0
- sentence-transformers: 2.5.0
- langchain/langchain-community: 0.2.15
- streamlit: 1.32.0
- pdfkit: 1.0.0
- wkhtmltopdf: 0.12.6 (system dependency)
- matplotlib: 3.9.2
- beautifulsoup4: 4.12.3
- bs4: 0.0.2
- selenium: 4.24.0
- webdriver-manager: 4.0.2
- PyPDF2: 3.0.1
- fpdf: 1.7.2
- radon: 6.0.1
- flake8: 7.1.1
- streamlit-extras: 0.3.6
- autogen: 0.2.35
- ag2: 0.2.35

### Architecture Changes

The setup now uses Poetry exclusively for dependency management, which provides:
- Better dependency resolution
- Cleaner environment isolation
- More reliable package installations

The placeholder groq_utils.py provides stub implementations to prevent import errors while allowing future integration with the Groq API if needed.