# pull_model.py
import streamlit as st

def pull_models():
    st.header("⬇ Pull a Model from Ollama Library")
    st.write("Enter the exact name of the model you want to pull from the Ollama library. You can just paste the whole model snippet from the model library page like 'ollama run llava-phi3' or you can just enter the model name like 'llava-phi3' and then click 'Pull Model' to begin the download. The progress of the download will be displayed below.")

    col1, col2 = st.columns([10, 1], vertical_alignment="bottom")

    with col1:
        model_name = st.text_input("Enter the name of the model you want to pull:")
    
    with col2:
        if st.button("Pull Model", key="pull_model"):
            if model_name:
                # Strip off "ollama run" or "ollama pull" from the beginning
                model_name = model_name.replace("ollama run ", "").replace("ollama pull ", "").strip()

                result = pull_model(model_name)
                if any("error" in status for status in result):
                    st.warning(f"Model '{model_name}' not found. Please make sure you've entered the correct model name. Model names often include a ':' to specify the variant. For example: 'mistral:instruct'")
                else:
                    for status in result:
                        st.write(status)
            else:
                st.error("Please enter a model name.")