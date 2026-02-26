import streamlit as st
import os
import sys

# Test script to verify our expander fix works

def main():
    st.title("Expander Fix Test")
    st.write("This tests whether our approach to avoiding nested expanders works")
    
    # First approach - container inside expander
    with st.expander("Method 1: Container Inside Expander", expanded=True):
        with st.container():
            st.write("This is inside a container within an expander")
            
            # Secondary UI elements that would have caused nesting issues
            st.subheader("Add Content")
            with st.form("form1"):
                content = st.text_area("Enter content:")
                submitted = st.form_submit_button("Submit")
            
            if submitted and content:
                st.success("Content added!")
    
    # Second approach - check expanded state
    expander2 = st.expander("Method 2: Check Expander State", expanded=True)
    
    if expander2.expanded:
        with expander2.container():
            st.write("This content only appears when the expander is expanded")
            st.subheader("Add More Content")
            with st.form("form2"):
                more_content = st.text_area("Enter more content:")
                more_submitted = st.form_submit_button("Submit")
            
            if more_submitted and more_content:
                st.success("More content added!")

if __name__ == "__main__":
    main()