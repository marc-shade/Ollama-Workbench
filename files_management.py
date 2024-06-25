# files_management.py
import streamlit as st
import os

def files_tab():
    st.subheader("Files")
    files_folder = "files"
    if not os.path.exists(files_folder):
        os.makedirs(files_folder)
    allowed_extensions = ['.json', '.txt', '.pdf', '.gif', '.jpg', '.jpeg', '.png']
    files = [f for f in os.listdir(files_folder) if os.path.isfile(os.path.join(files_folder, f)) and os.path.splitext(f)[1].lower() in allowed_extensions]

    for file in files:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(file)
        with col2:
            if file.endswith('.pdf'):
                st.button("📥", key=f"download_{file}")
            else:
                st.button("👁️", key=f"view_{file}")
        with col3:
            if not file.endswith('.pdf'):
                st.button("✏️", key=f"edit_{file}")
        with col4:
            st.button("🗑️", key=f"delete_{file}")

    for file in files:
        file_path = os.path.join(files_folder, file)
        
        if st.session_state.get(f"view_{file}", False):
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    file_content = f.read()
                st.text_area("File Content:", value=file_content, height=200, key=f"view_content_{file}")
            except UnicodeDecodeError:
                st.error(f"Unable to decode file {file}. It may be a binary file.")
        
        if st.session_state.get(f"edit_{file}", False):
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    file_content = f.read()
                new_content = st.text_area("Edit File Content:", value=file_content, height=200, key=f"edit_content_{file}")
                if st.button("Save Changes", key=f"save_{file}"):
                    with open(file_path, "w", encoding='utf-8') as f:
                        f.write(new_content)
                    st.success(f"Changes saved to {file}")
            except UnicodeDecodeError:
                st.error(f"Unable to decode file {file}. It may be a binary file.")
        
        if st.session_state.get(f"download_{file}", False):
            if file.endswith('.pdf'):
                with open(file_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=file,
                    mime='application/pdf',
                )
            else:
                with open(file_path, "r", encoding='utf-8') as f:
                    file_content = f.read()
                st.download_button(
                    label="Download File",
                    data=file_content,
                    file_name=file,
                    mime='text/plain',
                )
        
        if st.session_state.get(f"delete_{file}", False):
            os.remove(file_path)
            st.success(f"File {file} deleted.")
            st.experimental_rerun()
    

   # File upload section
    uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'json', 'gif', 'jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        file_path = os.path.join(files_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        st.experimental_rerun()