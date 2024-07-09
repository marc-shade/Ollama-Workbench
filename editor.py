# editor.py
import streamlit as st
import os
from pathlib import Path
import markdown
import bleach
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import re

def save_file(content, file_name, file_type):
    folder = "files"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, f"{file_name}.{file_type}")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def custom_markdown(content):
    def code_formatter(match):
        code = match.group(1)
        lexer = get_lexer_by_name("python", stripall=True)
        formatter = HtmlFormatter(style="monokai")
        return highlight(code, lexer, formatter)

    content = re.sub(r'```python\n(.*?)\n```', code_formatter, content, flags=re.DOTALL)
    html = markdown.markdown(content, extensions=['fenced_code', 'codehilite'])
    return bleach.clean(html, tags=['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'code', 'pre', 'span'], attributes={'span': ['class']})

def insert_text(text):
    if 'editor_content' not in st.session_state:
        st.session_state.editor_content = ""
    st.session_state.editor_content += text

def main():
    st.title("Enhanced Rich Text Editor")

    # File operations
    col1, col2, col3 = st.columns(3)
    with col1:
        file_name = st.text_input("File name", "document")
    with col2:
        file_type = st.selectbox("File type", ["md", "txt"])
    with col3:
        files = [f for f in os.listdir("files") if f.endswith(f".{file_type}")]
        selected_file = st.selectbox("Load file", [""] + files)

    # Markdown formatting buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Bold"):
            insert_text("**Bold**")
    with col2:
        if st.button("Italic"):
            insert_text("*Italic*")
    with col3:
        if st.button("List Item"):
            insert_text("\n- List item")
    with col4:
        if st.button("Code Block"):
            insert_text("\n```python\n# Your code here\n```\n")
    with col5:
        if st.button("Link"):
            insert_text("[Link text](https://example.com)")

    # Editor
    if 'editor_content' not in st.session_state:
        st.session_state.editor_content = ""
    
    editor_content = st.text_area("Editor", st.session_state.editor_content, height=300, key="editor")
    st.session_state.editor_content = editor_content

    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save"):
            file_path = save_file(editor_content, file_name, file_type)
            st.success(f"File saved: {file_path}")
    
    with col2:
        if st.button("Load"):
            if selected_file:
                file_path = os.path.join("files", selected_file)
                st.session_state.editor_content = load_file(file_path)
                st.experimental_rerun()
            else:
                st.warning("Please select a file to load.")

    with col3:
        if st.button("Clear"):
            st.session_state.editor_content = ""
            st.experimental_rerun()

    # Preview
    if editor_content:
        st.subheader("Preview")
        if file_type == "md":
            styled_html = f"""
            <style>
                {HtmlFormatter(style="monokai").get_style_defs('.highlight')}
            </style>
            {custom_markdown(editor_content)}
            """
            st.markdown(styled_html, unsafe_allow_html=True)
        else:
            st.text(editor_content)

if __name__ == "__main__":
    main()