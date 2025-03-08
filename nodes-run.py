import multiprocessing
import streamlit as st

# Function to start the Quart app
def start_quart_app():
    import os
    os.system('python nodes.py')

if __name__ == "__main__":
    # Start the Quart app in a separate process
    p = multiprocessing.Process(target=start_quart_app)
    p.start()

    # Your Streamlit code goes here
    st.title('Streamlit and Quart Integration')
    st.write('The Quart app is running in the background.')

    # Ensure the Quart process is terminated when Streamlit stops
    p.join()