import streamlit as st
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from chat_workspace import chat_workspace_ui, save_ai_content_to_workspace

# Test content
test_content = '''
Here's an example of how to implement a simple web server in Python:

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello_world():
    return jsonify({"message": "Hello, World!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Title: Building a RESTful API

RESTful APIs follow these key principles:
1. Client-Server architecture
2. Statelessness
3. Cacheability
4. Layered system
5. Uniform interface

Make sure your API endpoints use proper HTTP methods (GET, POST, PUT, DELETE) for CRUD operations.
'''

def main():
    st.title("Chat Workspace Component Test")
    st.markdown("This is a test of the chat workspace component to validate that the nested expander issue is fixed.")
    
    # Button to add test content
    if st.button("Add Test Content"):
        save_ai_content_to_workspace(test_content)
        st.success("Test content added to workspace")
    
    # Outer container - simulating the main chat interface structure
    st.header("Workspace Section")
    with st.expander("Workspace", expanded=True):
        # This should now work without the nested expander error
        chat_workspace_ui()

if __name__ == "__main__":
    main()