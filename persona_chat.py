import streamlit as st
import random
from datetime import datetime
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import requests
from PIL import Image
import io
import base64
import hashlib
from ollama_utils import call_ollama_endpoint

@dataclass
class Persona:
    name: str
    age: int
    nationality: str
    occupation: str
    background: str
    routine: str
    personality: str
    skills: List[str]
    avatar: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 150

# Predefined occupations for quick persona generation
STARTER_OCCUPATIONS = [
    "Data Scientist",
    "Software Engineer",
    "Doctor",
    "Artist",
    "Teacher",
    "Entrepreneur",
    "Writer",
    "Chef"
]

def generate_avatar(persona_name: str) -> str:
    """Generate a unique avatar based on the persona name."""
    # Create a hash of the name to ensure consistent but unique avatars
    name_hash = hashlib.md5(persona_name.encode()).hexdigest()
    
    # Use the hash to generate avatar parameters
    colors = ['FF5733', '33FF57', '3357FF', 'FF33F6', 'F6FF33']
    color = colors[int(name_hash[:2], 16) % len(colors)]
    
    # Generate a unique avatar URL using DiceBear API
    avatar_url = f"https://api.dicebear.com/7.x/personas/svg?seed={name_hash}&backgroundColor={color}"
    return avatar_url

def create_persona(occupation: str = None) -> Persona:
    """Create a new persona with optional occupation specification."""
    if occupation is None:
        occupation = random.choice(STARTER_OCCUPATIONS)
    
    # Use Ollama to generate persona details
    prompt = f"""You are a creative AI assistant specializing in creating detailed, realistic personas. Generate a complete persona for a {occupation}.

Your response must be a valid JSON object with exactly this format:
{{
    "name": "<full name - be diverse and realistic>",
    "age": <number between 25-65>,
    "nationality": "<country - be diverse in your choices>",
    "background": "<detailed 2-3 sentence professional background including education and career progression>",
    "routine": "<detailed daily routine from morning to evening, including work and personal life>",
    "personality": "<detailed description of personality traits, communication style, and work approach>",
    "skills": [
        "<specific technical skill relevant to their occupation>",
        "<specific soft skill that defines their work style>",
        "<unique or interesting skill that makes them stand out>"
    ]
}}

Make the persona feel like a real person with:
- A coherent and believable background story
- A realistic daily routine that matches their profession
- Personality traits that feel authentic and three-dimensional
- Skills that are specific and relevant to their role

Respond ONLY with the JSON object, no other text."""
    
    try:
        # Use the proper call_ollama_endpoint function
        response, _, _, _ = call_ollama_endpoint(
            model="mistral:instruct",
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000,
            presence_penalty=0.3,
            frequency_penalty=0.3
        )
        
        # Clean up the response to ensure valid JSON
        json_str = response.strip()
        # Remove any markdown code block indicators
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        persona_data = json.loads(json_str)
        
        # Create and return persona
        return Persona(
            name=persona_data["name"],
            age=persona_data["age"],
            nationality=persona_data["nationality"],
            occupation=occupation,
            background=persona_data["background"],
            routine=persona_data["routine"],
            personality=persona_data["personality"],
            skills=persona_data["skills"],
            avatar=generate_avatar(persona_data["name"]),
            model="mistral:instruct"
        )
    except json.JSONDecodeError as e:
        st.error(f"Error parsing persona data: {str(e)}\nResponse: {json_str}")
        return None
    except KeyError as e:
        st.error(f"Missing required field in persona data: {str(e)}\nResponse: {json_str}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def save_personas(personas: List[Persona]):
    """Save personas to a JSON file."""
    personas_data = [vars(p) for p in personas]
    os.makedirs("personas", exist_ok=True)
    with open("personas/saved_personas.json", "w") as f:
        json.dump(personas_data, f, indent=2)

def load_personas() -> List[Persona]:
    """Load personas from a JSON file."""
    try:
        with open("personas/saved_personas.json", "r") as f:
            personas_data = json.load(f)
            return [Persona(**p) for p in personas_data]
    except FileNotFoundError:
        return []

def persona_group_chat():
    st.title("Persona Group Chat")
    
    # Add link to Persona Lab
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Create conversations between AI personas with different backgrounds and personalities.")
    with col2:
        if st.button("📝 Manage Personas", help="Open Persona Lab to manage all personas"):
            st.session_state.selected_test = "Persona Lab"
            st.rerun()
    
    # Initialize session state
    if "personas" not in st.session_state:
        st.session_state.personas = load_personas()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for persona management
    with st.sidebar:
        st.subheader("Manage Personas")
        
        # Quick add buttons for starter occupations
        st.write("Quick Add by Occupation:")
        cols = st.columns(2)
        for i, occupation in enumerate(STARTER_OCCUPATIONS):
            if cols[i % 2].button(occupation):
                new_persona = create_persona(occupation)
                if new_persona:
                    st.session_state.personas.append(new_persona)
                    save_personas(st.session_state.personas)
        
        # Manual persona creation
        st.write("---")
        st.write("Custom Persona:")
        with st.form("create_persona"):
            name = st.text_input("Name")
            age = st.number_input("Age", 25, 65, 30)
            nationality = st.text_input("Nationality")
            occupation = st.text_input("Occupation")
            model = st.selectbox("Model", ["mistral:instruct", "llama2", "codellama", "neural-chat"])
            
            if st.form_submit_button("Create Custom Persona"):
                if name and nationality and occupation:
                    new_persona = Persona(
                        name=name,
                        age=age,
                        nationality=nationality,
                        occupation=occupation,
                        background="",
                        routine="",
                        personality="",
                        skills=[],
                        avatar=generate_avatar(name),
                        model=model
                    )
                    st.session_state.personas.append(new_persona)
                    save_personas(st.session_state.personas)
    
    # Main chat area
    if not st.session_state.personas:
        st.info("Add some personas using the sidebar to start the group chat!")
        return
    
    # Display current personas
    st.subheader("Current Personas")
    cols = st.columns(len(st.session_state.personas))
    for i, persona in enumerate(st.session_state.personas):
        with cols[i]:
            st.image(persona.avatar, width=100)
            st.write(f"**{persona.name}**")
            st.write(f"*{persona.occupation}*")
            if st.button(f"Remove {persona.name}", key=f"remove_{i}"):
                st.session_state.personas.pop(i)
                save_personas(st.session_state.personas)
                st.rerun()
    
    # Chat interface
    st.write("---")
    st.subheader("Group Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.write(f"**{message.get('name', 'User')}:** {message['content']}")
    
    # User input
    if prompt := st.chat_input("Enter your message"):
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "name": "User"
        })
        
        # Get responses from all personas
        for persona in st.session_state.personas:
            context = f"""You are {persona.name}, a {persona.age}-year-old {persona.nationality} {persona.occupation}.
            Background: {persona.background}
            Personality: {persona.personality}
            
            Previous conversation:
            {chr(10).join([f"{m['name']}: {m['content']}" for m in st.session_state.chat_history[-5:]])}
            
            Respond as {persona.name}, keeping in mind your background and personality.
            Keep the response concise (max 2-3 sentences).
            """
            
            try:
                response, _, _, _ = call_ollama_endpoint(
                    model=persona.model,
                    prompt=context,
                    temperature=persona.temperature,
                    max_tokens=persona.max_tokens
                )
                
                # Add persona's response to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "name": persona.name,
                    "avatar": persona.avatar
                })
            except Exception as e:
                st.error(f"Error getting response from {persona.name}: {str(e)}")
        
        st.rerun()

if __name__ == "__main__":
    persona_group_chat()
