import streamlit as st
import uuid
import json
from datetime import datetime
from typing import Optional
from persona_lab.persona_model import Persona, PersonaDB
from ollama_utils import call_ollama_endpoint
from persona_chat import STARTER_OCCUPATIONS

class PersonaLab:
    def __init__(self):
        self.db = PersonaDB()
        if "editing_persona" not in st.session_state:
            st.session_state.editing_persona = None
        if "show_history" not in st.session_state:
            st.session_state.show_history = False
        if "current_tab" not in st.session_state:
            st.session_state.current_tab = "Browse"

    def generate_persona(self, occupation: str) -> Optional[Persona]:
        """Generate a new persona with the given occupation."""
        prompt = f"""You are a creative AI assistant specializing in creating detailed, realistic personas. Generate a complete persona for a {occupation}.

Your response must be a valid JSON object with exactly this format:
{{
    "name": "<full name - be diverse and realistic>",
    "age": <number between 25-65>,
    "nationality": "<country - be diverse in your choices>",
    "background": "<detailed 3-4 sentence professional background including education, career progression, and key achievements>",
    "routine": "<detailed daily routine from morning to evening, including work habits, breaks, and personal life>",
    "personality": "<detailed description of personality traits, communication style, work approach, and unique characteristics>",
    "skills": [
        "<specific technical skill relevant to their occupation with proficiency level>",
        "<specific soft skill that defines their work style with examples>",
        "<unique or interesting skill that makes them stand out with context>"
    ]
}}

Make the persona feel like a real person with:
- A coherent and believable background story that shows career progression
- A detailed daily routine that reflects their profession and personality
- Rich personality traits that affect their work and communication style
- Specific and measurable skills that match their experience level

Respond ONLY with the JSON object, no other text."""

        try:
            response, _, _, _ = call_ollama_endpoint(
                model="mistral:instruct",
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500,
                presence_penalty=0.3,
                frequency_penalty=0.3
            )

            # Clean and parse response
            json_str = response.strip().replace("```json", "").replace("```", "").strip()
            persona_data = json.loads(json_str)

            # Create new persona
            persona = Persona(
                id=str(uuid.uuid4()),
                name=persona_data["name"],
                age=persona_data["age"],
                nationality=persona_data["nationality"],
                occupation=occupation,
                background=persona_data["background"],
                routine=persona_data["routine"],
                personality=persona_data["personality"],
                skills=persona_data["skills"],
                avatar=self.generate_avatar(persona_data["name"]),
                model="mistral:instruct",
                created_at=datetime.now(),
                modified_at=datetime.now(),
                version=1,
                generated_by="AI"
            )

            # Save to database
            if self.db.create_persona(persona):
                return persona
            return None
        except Exception as e:
            st.error(f"Error generating persona: {str(e)}")
            return None

    def edit_persona(self, persona_id: str):
        """Show the persona edit form."""
        persona = self.db.get_persona(persona_id)
        if not persona:
            st.error("Persona not found")
            return

        with st.form(f"edit_persona_{persona_id}"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name", persona.name)
                age = st.number_input("Age", 25, 65, persona.age)
                nationality = st.text_input("Nationality", persona.nationality)
                occupation = st.text_input("Occupation", persona.occupation)
                model = st.selectbox("Model", ["mistral:instruct", "llama2", "codellama", "neural-chat"], 
                                   index=["mistral:instruct", "llama2", "codellama", "neural-chat"].index(persona.model))
                temperature = st.slider("Temperature", 0.0, 1.0, persona.temperature)
                max_tokens = st.number_input("Max Tokens", 100, 2000, persona.max_tokens)

            with col2:
                background = st.text_area("Background", persona.background)
                routine = st.text_area("Daily Routine", persona.routine)
                personality = st.text_area("Personality", persona.personality)
                skills = st.text_area("Skills (one per line)", "\n".join(persona.skills))
                notes = st.text_area("Notes", persona.notes)
                tags = st.multiselect("Tags", self.db.get_all_tags(), persona.tags)

            if st.form_submit_button("Save Changes"):
                updated_persona = Persona(
                    id=persona.id,
                    name=name,
                    age=age,
                    nationality=nationality,
                    occupation=occupation,
                    background=background,
                    routine=routine,
                    personality=personality,
                    skills=skills.split("\n"),
                    avatar=persona.avatar,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    created_at=persona.created_at,
                    modified_at=datetime.now(),
                    version=persona.version,
                    tags=tags,
                    notes=notes,
                    generated_by=persona.generated_by,
                    interaction_history=persona.interaction_history,
                    metadata=persona.metadata
                )
                if self.db.update_persona(updated_persona, modified_by="user"):
                    st.success("Changes saved successfully!")
                    st.rerun()

    def show_persona_details(self, persona: Persona):
        """Show detailed view of a persona."""
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(persona.avatar, width=150)
            st.write("---")
            # Action buttons in a more compact layout
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("✏️ Edit", key=f"edit_{persona.id}"):
                    st.session_state.editing_persona = persona.id
                if st.button("🗑️ Delete", key=f"delete_{persona.id}", type="secondary"):
                    st.session_state.deleting_persona = persona.id
            with btn_col2:
                if st.button("📋 History", key=f"history_{persona.id}"):
                    st.session_state.show_history = persona.id
                if st.button("🔄 Regen", key=f"regen_{persona.id}"):
                    st.session_state.regenerating_persona = persona.id

        with col2:
            # Header with metadata
            st.subheader(f"{persona.name} ({persona.age})")
            st.caption(f"{persona.nationality} · {persona.occupation}")
            
            # Model settings in a compact format
            with st.expander("🛠️ Model Configuration", expanded=False):
                st.write(f"**Model:** {persona.model}")
                st.write(f"**Temperature:** {persona.temperature}")
                st.write(f"**Max Tokens:** {persona.max_tokens}")
            
            # Main content in tabs
            tabs = st.tabs(["📝 Details", "📅 Routine", "🎭 Personality", "🎯 Skills"])
            
            with tabs[0]:
                st.markdown("### Background")
                st.write(persona.background)
                if persona.notes:
                    st.markdown("### Notes")
                    st.write(persona.notes)
                if persona.tags:
                    st.markdown("### Tags")
                    for tag in persona.tags:
                        st.caption(f"#{tag}")
            
            with tabs[1]:
                st.write(persona.routine)
            
            with tabs[2]:
                st.write(persona.personality)
            
            with tabs[3]:
                for skill in persona.skills:
                    st.markdown(f"- {skill}")

        # Show edit form if this persona is being edited
        if getattr(st.session_state, 'editing_persona', None) == persona.id:
            st.markdown("---")
            st.subheader("Edit Persona")
            self.show_edit_form(persona)

        # Show delete confirmation if this persona is being deleted
        if getattr(st.session_state, 'deleting_persona', None) == persona.id:
            st.markdown("---")
            st.warning("Are you sure you want to delete this persona?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", type="primary"):
                    if self.db.delete_persona(persona.id):
                        st.success("Persona deleted successfully!")
                        st.session_state.deleting_persona = None
                        st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.deleting_persona = None
                    st.rerun()

        # Show history if requested
        if getattr(st.session_state, 'show_history', None) == persona.id:
            st.markdown("---")
            st.subheader("Change History")
            history = self.db.get_persona_history(persona.id)
            if not history:
                st.info("No changes recorded yet")
            else:
                for entry in history:
                    with st.expander(f"{entry.field_name} - {entry.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Previous Value:**")
                            st.text(entry.old_value)
                        with col2:
                            st.markdown("**New Value:**")
                            st.text(entry.new_value)
                        st.caption(f"Modified by: {entry.modified_by}")

    def show_edit_form(self, persona: Persona):
        """Show the persona edit form."""
        with st.form(f"edit_persona_{persona.id}"):
            tabs = st.tabs(["Basic Info", "Details", "Model Settings", "Tags & Notes"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Name", persona.name)
                    age = st.number_input("Age", 25, 65, persona.age)
                    nationality = st.text_input("Nationality", persona.nationality)
                with col2:
                    occupation = st.text_input("Occupation", persona.occupation)
                    avatar = st.text_input("Avatar URL (optional)", persona.avatar)
            
            with tabs[1]:
                background = st.text_area("Background", persona.background, height=150)
                routine = st.text_area("Daily Routine", persona.routine, height=150)
                personality = st.text_area("Personality", persona.personality, height=150)
                skills = st.text_area("Skills (one per line)", "\n".join(persona.skills), height=100)
            
            with tabs[2]:
                col1, col2 = st.columns(2)
                with col1:
                    model = st.selectbox("Model", 
                                       ["mistral:instruct", "llama2", "codellama", "neural-chat"],
                                       index=["mistral:instruct", "llama2", "codellama", "neural-chat"].index(persona.model))
                    temperature = st.slider("Temperature", 0.0, 1.0, persona.temperature)
                with col2:
                    max_tokens = st.number_input("Max Tokens", 100, 2000, persona.max_tokens)
            
            with tabs[3]:
                available_tags = self.db.get_all_tags()
                tags = st.multiselect("Tags", available_tags, persona.tags)
                new_tag = st.text_input("Add New Tag")
                notes = st.text_area("Notes", persona.notes)

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Save Changes", type="primary"):
                    # Add new tag if provided
                    if new_tag:
                        self.db.add_tag(new_tag)
                        tags.append(new_tag)
                    
                    # Update persona
                    updated_persona = Persona(
                        id=persona.id,
                        name=name,
                        age=age,
                        nationality=nationality,
                        occupation=occupation,
                        background=background,
                        routine=routine,
                        personality=personality,
                        skills=skills.split("\n") if skills else [],
                        avatar=avatar or persona.avatar,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        created_at=persona.created_at,
                        modified_at=datetime.now(),
                        version=persona.version,
                        tags=tags,
                        notes=notes,
                        generated_by=persona.generated_by,
                        interaction_history=persona.interaction_history,
                        metadata=persona.metadata
                    )
                    
                    if self.db.update_persona(updated_persona, modified_by="user"):
                        st.success("Changes saved successfully!")
                        st.session_state.editing_persona = None
                        st.rerun()
            with col2:
                if st.form_submit_button("Cancel", type="secondary"):
                    st.session_state.editing_persona = None
                    st.rerun()

    def delete_persona(self, persona_id: str):
        """Delete a persona with confirmation."""
        if st.button("Confirm Delete"):
            if self.db.delete_persona(persona_id):
                st.success("Persona deleted successfully!")
                st.rerun()
            else:
                st.error("Error deleting persona")

    def show_history(self, persona_id: str):
        """Show the history of changes for a persona."""
        history = self.db.get_persona_history(persona_id)
        if not history:
            st.info("No history available for this persona")
            return

        for entry in history:
            with st.expander(f"Changed {entry.field_name} on {entry.timestamp}"):
                st.write("**Old Value:**", entry.old_value)
                st.write("**New Value:**", entry.new_value)
                st.write("**Modified By:**", entry.modified_by)

    def show_interface(self):
        """Main interface for the Persona Lab."""
        st.title("🧪 Persona Lab")
        
        # Tabs for different sections
        tabs = st.tabs(["Browse", "Create", "Search", "Analytics"])
        
        with tabs[0]:  # Browse
            personas = self.db.get_all_personas()
            if not personas:
                st.info("No personas created yet. Create one in the Create tab!")
            else:
                for persona in personas:
                    with st.expander(f"{persona.name} - {persona.occupation}"):
                        self.show_persona_details(persona)

        with tabs[1]:  # Create
            st.subheader("Create New Persona")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Quick Add")
                for occupation in STARTER_OCCUPATIONS:
                    if st.button(occupation):
                        with st.spinner(f"Generating {occupation.lower()} persona..."):
                            if self.generate_persona(occupation):
                                st.success(f"Created new {occupation.lower()} persona!")
                                st.rerun()
            
            with col2:
                st.write("Custom Persona")
                with st.form("create_custom_persona"):
                    # Form fields for manual persona creation
                    # (Similar to edit_persona form)
                    pass

        with tabs[2]:  # Search
            st.subheader("Search Personas")
            query = st.text_input("Search by name, occupation, or any other field")
            if query:
                results = self.db.search_personas(query)
                if results:
                    for persona in results:
                        with st.expander(f"{persona.name} - {persona.occupation}"):
                            self.show_persona_details(persona)
                else:
                    st.info("No matching personas found")

        with tabs[3]:  # Analytics
            st.subheader("Persona Analytics")
            # Add visualization of persona data, trends, etc.
            pass

def persona_lab_interface():
    lab = PersonaLab()
    lab.show_interface()
