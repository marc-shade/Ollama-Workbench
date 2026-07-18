"""
Voice interface for Ollama Workbench.
Provides UI components for voice input/output and voice settings management.
"""
import streamlit as st
import ollama_workbench.chat.voice_utils as voice_utils

import os
import json
import time
from datetime import datetime

def voice_settings_ui():
    """UI for managing voice settings."""
    st.title("Voice Settings")
    
    # Get available voices
    available_voices = voice_utils.get_available_voices()
    
    # Create tabs for voice profiles
    profile_tabs = st.tabs(["Voice Profiles", "Add New Profile"])
    
    with profile_tabs[0]:
        st.subheader("Manage Voice Profiles")
        
        # Voice profiles table
        profiles_data = []
        for voice_name in available_voices:
            settings = voice_utils.get_voice_settings(voice_name)
            profiles_data.append({
                "Profile Name": voice_name,
                "Provider": settings.get('provider', 'gtts'),
                "Voice ID": settings.get('voice_id', 'default'),
                "Language": settings.get('language', 'en')
            })
        
        if profiles_data:
            st.dataframe(profiles_data)
        else:
            st.info("No voice profiles found. Create a new one.")
        
        # Voice profile management
        selected_profile = st.selectbox("Select Profile to Manage", available_voices, key="profile_selector")
        if selected_profile:
            settings = voice_utils.get_voice_settings(selected_profile)
            
            st.subheader(f"Edit {selected_profile} Profile")
            
            # Cannot delete the default profile
            if selected_profile != "default" and st.button("🗑️ Delete Profile", key=f"delete_{selected_profile}"):
                if voice_utils.remove_voice_profile(selected_profile):
                    st.success(f"Profile '{selected_profile}' deleted successfully")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Failed to delete profile '{selected_profile}'")
            
            provider = st.selectbox(
                "Voice Provider",
                ["gtts", "elevenlabs", "tts_server"],
                index=["gtts", "elevenlabs", "tts_server"].index(settings.get('provider', 'gtts')),
                key="provider_selector_edit"
            )
            
            language = st.text_input("Language Code", value=settings.get('language', 'en'), key="language_edit")
            
            voice_id = st.text_input("Voice ID", value=settings.get('voice_id', 'en-US-Wavenet-D'), key="voice_id_edit")
            
            speed = st.slider(
                "Speech Speed",
                min_value=0.5,
                max_value=2.0,
                value=settings.get('speed', 1.0),
                step=0.1,
                key="speed_edit"
            )
            
            pitch = st.slider(
                "Voice Pitch",
                min_value=0.5,
                max_value=2.0,
                value=settings.get('pitch', 1.0),
                step=0.1,
                key="pitch_edit"
            )
            
            # Test voice
            test_text = st.text_input("Test Text", value="This is a test of the voice settings.", key="test_text_edit")
            
            if st.button("🔊 Test Voice"):
                updated_settings = {
                    'provider': provider,
                    'language': language,
                    'voice_id': voice_id,
                    'speed': speed,
                    'pitch': pitch
                }
                
                # Temporarily update the profile
                voice_utils.add_voice_profile(selected_profile, updated_settings)
                
                # Generate and play speech
                audio_file = voice_utils.text_to_speech(test_text, selected_profile)
                
                if audio_file:
                    voice_utils.play_speech(audio_file)
                    st.success("Voice test completed.")
                else:
                    st.error("Failed to generate speech.")
            
            # Save changes
            if st.button("💾 Save Changes"):
                updated_settings = {
                    'provider': provider,
                    'language': language,
                    'voice_id': voice_id,
                    'speed': speed,
                    'pitch': pitch
                }
                
                if voice_utils.add_voice_profile(selected_profile, updated_settings):
                    st.success(f"Profile '{selected_profile}' updated successfully")
                else:
                    st.error(f"Failed to update profile '{selected_profile}'")
    
    with profile_tabs[1]:
        st.subheader("Add New Voice Profile")
        
        new_profile_name = st.text_input("Profile Name", key="profile_name_new")
        new_provider = st.selectbox("Voice Provider", ["gtts", "elevenlabs", "tts_server"], key="provider_selector_new")
        new_language = st.text_input("Language Code", value="en", key="language_new")
        new_voice_id = st.text_input("Voice ID", value="en-US-Wavenet-D", key="voice_id_new")
        new_speed = st.slider("Speech Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="speed_new")
        new_pitch = st.slider("Voice Pitch", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="pitch_new")
        
        # Create new profile
        if st.button("➕ Create Profile") and new_profile_name:
            new_settings = {
                'provider': new_provider,
                'language': new_language,
                'voice_id': new_voice_id,
                'speed': new_speed,
                'pitch': new_pitch
            }
            
            if voice_utils.add_voice_profile(new_profile_name, new_settings):
                st.success(f"Profile '{new_profile_name}' created successfully")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(f"Failed to create profile '{new_profile_name}'")

def voice_input_component(callback=None, placeholder_text="Click to speak..."):
    """Voice input component for streamlit UI."""
    col1, col2 = st.columns([1, 10])
    
    with col1:
        is_listening = st.session_state.get("is_listening", False)
        button_label = "🛑" if is_listening else "🎤"
        button_help = "Stop listening" if is_listening else "Start listening"
        
        if st.button(button_label, help=button_help, key="voice_input_button"):
            if is_listening:
                st.session_state.is_listening = False
                # Stop listening and get the text
                text = voice_utils.stop_voice_input()
                
                if callback and text:
                    callback(text)
                
                st.rerun()
            else:
                st.session_state.is_listening = True
                
                # Define callback for real-time transcription
                def speech_callback(text):
                    if callback:
                        callback(text)
                    st.session_state.is_listening = False
                    st.rerun()
                
                def error_callback(error):
                    st.error(f"Voice input error: {error}")
                    st.session_state.is_listening = False
                    st.rerun()
                
                # Start listening
                voice_utils.start_voice_input(speech_callback, error_callback)
                st.rerun()
    
    with col2:
        if is_listening:
            st.info("🎙️ Listening... Speak now")
        else:
            st.text_input("Voice input", placeholder=placeholder_text, key="voice_input_text_display", disabled=True)

def voice_chat_interface():
    """Complete voice chat interface."""
    st.title("Voice Chat")
    
    if "voice_chat_history" not in st.session_state:
        st.session_state.voice_chat_history = []
    
    # Display chat history
    for i, msg in enumerate(st.session_state.voice_chat_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Add play button for assistant messages
            if msg["role"] == "assistant":
                if st.button("🔊", key=f"play_msg_{i}"):
                    audio_file = voice_utils.text_to_speech(msg["content"])
                    if audio_file:
                        voice_utils.play_speech(audio_file)
    
    # Voice input component
    def on_voice_input(text):
        if text:
            # Add user message to history
            st.session_state.voice_chat_history.append({
                "role": "user",
                "content": text
            })
            
            # Process the request and get response using Ollama or other models
            try:
                # Get the selected model from session state or use a default
                selected_model = st.session_state.get("voice_chat_model", "llama3")
                
                # Determine the provider; un-prefixed model names are local Ollama models
                provider = "ollama"
                if "🦙 Ollama Models" in selected_model:
                    selected_model = selected_model.replace("🦙 Ollama Models ", "")
                    provider = "ollama"
                elif "🚀 Groq Models" in selected_model:
                    selected_model = selected_model.replace("🚀 Groq Models ", "")
                    provider = "groq"
                elif "🤖 OpenAI Models" in selected_model:
                    selected_model = selected_model.replace("🤖 OpenAI Models ", "")
                    provider = "openai"
                elif "🌟 Mistral Models" in selected_model:
                    selected_model = selected_model.replace("🌟 Mistral Models ", "")
                    provider = "mistral"
                
                # Get response based on provider
                from ollama_workbench.providers.ollama_utils import call_ollama_endpoint
                from ollama_workbench.providers.openai_utils import call_openai_api
                from ollama_workbench.providers.groq_utils import call_groq_api
                from ollama_workbench.providers.mistral_utils import call_mistral_api
                
                # Get model parameters
                temperature = st.session_state.get("voice_chat_temperature", 0.7)
                max_tokens = st.session_state.get("voice_chat_max_tokens", 500)
                
                # Process previous conversation history for context
                context = []
                for msg in st.session_state.voice_chat_history:
                    context.append({"role": msg["role"], "content": msg["content"]})
                
                # Add current message to context
                context.append({"role": "user", "content": text})
                
                # Generate response based on provider
                if provider == "ollama":
                    with st.spinner("Generating response..."):
                        response, _, _, _, _ = call_ollama_endpoint(
                            model=selected_model,
                            prompt=text,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                elif provider == "openai":
                    with st.spinner("Generating response..."):
                        response_data = call_openai_api(
                            model=selected_model,
                            prompt=context,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        response = response_data['choices'][0]['message']['content']
                elif provider == "groq":
                    with st.spinner("Generating response..."):
                        response_data = call_groq_api(
                            model=selected_model,
                            prompt=context,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        response = response_data['choices'][0]['message']['content']
                elif provider == "mistral":
                    with st.spinner("Generating response..."):
                        response_data = call_mistral_api(
                            model=selected_model,
                            prompt=context,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        response = response_data['choices'][0]['message']['content']
                else:
                    # Default fallback to Ollama if provider not recognized
                    with st.spinner("Generating response..."):
                        response, _, _, _, _ = call_ollama_endpoint(
                            model="llama3",
                            prompt=text,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
            except Exception as e:
                response = f"I encountered an error while processing your request: {str(e)}"
                st.error(f"Error generating response: {str(e)}")
            
            # Add assistant response to history
            st.session_state.voice_chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Auto-play response
            audio_file = voice_utils.text_to_speech(response)
            if audio_file:
                voice_utils.play_speech(audio_file)
    
    voice_input_component(on_voice_input, "Click the microphone icon to speak")
    
    # Text input as backup
    user_input = st.chat_input("Type your message here...")
    if user_input:
        on_voice_input(user_input)
        st.rerun()
    
    # Voice and model settings
    with st.expander("Settings", expanded=False):
        tabs = st.tabs(["Voice Settings", "Model Settings"])
        
        with tabs[0]:
            st.subheader("Voice Settings")
            voices = voice_utils.get_available_voices()
            selected_voice = st.selectbox("Voice Profile", voices, key="voice_profile_settings")
            
            # Test selected voice
            test_text = st.text_input("Test Text", value="This is a test of the selected voice profile.", key="test_text_settings")
            if st.button("🔊 Test", key="test_button_settings"):
                audio_file = voice_utils.text_to_speech(test_text, selected_voice)
                if audio_file:
                    voice_utils.play_speech(audio_file)
        
        with tabs[1]:
            st.subheader("Model Settings")
            
            # Import models from ollama_utils
            from ollama_workbench.providers.ollama_utils import get_all_models
            
            # Get available models
            all_models = get_all_models()
            
            # Model selection
            selected_model = st.selectbox(
                "Select Model", 
                all_models,
                index=1 if len(all_models) > 1 else 0,  # Default to first actual model
                key="voice_chat_model"
            )
            
            # Temperature setting
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="voice_chat_temperature",
                help="Higher values make output more random, lower values more deterministic"
            )
            
            # Max tokens setting
            max_tokens = st.slider(
                "Max Tokens",
                min_value=50,
                max_value=2000,
                value=500,
                step=50,
                key="voice_chat_max_tokens",
                help="Maximum number of tokens to generate"
            )

if __name__ == "__main__":
    st.set_page_config(
        page_title="Voice Interface",
        page_icon="🎤",
        layout="wide"
    )
    
    tab1, tab2 = st.tabs(["Voice Chat", "Voice Settings"])
    
    with tab1:
        voice_chat_interface()
    
    with tab2:
        voice_settings_ui()