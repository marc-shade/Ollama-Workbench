# info_brainstorm.py
import streamlit as st

def display_info_brainstorm():
    with st.expander("❓ About Brainstorm", expanded=False):
        st.html("""<img src='https://2acrestudios.com/wp-content/uploads/2024/05/00017-1652154938.png'  style='max-width: 200px;' />""")
        st.markdown(
        """
        ## How It Works
        
        Brainstorm is an advanced collaborative AI feature within Ollama Workbench that allows users to create, manage, and interact with multiple AI agents in a structured workflow. Here's how it works:

        1. **Agent Management**: Users can create and customize multiple AI agents, each with unique characteristics such as:
           - Name and emoji for easy identification
           - Underlying language model
           - Specialized roles (e.g., creative writer, code expert, analyst)
           - Personality traits and communication styles
           - Knowledge bases or specific data sets to draw from

        2. **Workflow Creation**: Users can design workflows by selecting a sequence of agents to respond to prompts. This allows for a multi-perspective approach to problem-solving or idea generation.

        3. **Interactive Sessions**: During a brainstorming session, users can input prompts or questions, and the selected agents will respond in the predetermined sequence, building upon each other's ideas and insights.

        4. **Workflow Management**: Users can save, load, and modify workflows, making it easy to reuse effective agent combinations for different projects or topics.

        5. **Dynamic Adjustment**: During a session, users can adjust the number of agents, change the sequence, or modify agent settings to optimize the brainstorming process.

        6. **Conversation History**: The system maintains a detailed conversation history, allowing users to review and analyze the collaborative thought process.

        ## Example Use Case: Product Innovation Workshop

        Imagine a product development team at a tech startup using Brainstorm to generate ideas for a new smart home device.

        ### Setup:

        1. The team creates the following agents:
           - 🐱 Mia the Creative: A free-thinking idea generator
           - 🤖 Codi the Coder: An expert in IoT and embedded systems
           - 🦉 Rev the Reviewer: A critical thinker and market analyst
           - 🐙 Otto the Optimizer: A specialist in user experience and product optimization
           - 🦊 Fin the Consultant: An industry expert with knowledge of current smart home trends

        2. They create a workflow with the sequence: Mia → Codi → Rev → Otto → Fin

        ### Brainstorming Session:

        1. The team inputs the prompt: "Propose an innovative smart home device that addresses a common household problem."

        2. Mia the Creative suggests: "A smart plant care system that automatically waters plants, adjusts lighting, and provides health diagnostics."

        3. Codi the Coder builds on this: "We can use soil moisture sensors, LED grow lights, and a Raspberry Pi to control the system. I can outline the basic architecture and components needed."

        4. Rev the Reviewer analyzes: "The idea has potential, but we need to consider market saturation. Let's focus on unique features like plant species recognition and customized care plans."

        5. Otto the Optimizer suggests: "To enhance user experience, we could add a mobile app with AI-powered plant recognition. Users can snap a photo of their plant to get instant care instructions and integration with the smart system."

        6. Fin the Consultant provides industry context: "This aligns well with the growing trend of indoor gardening and sustainability. To stand out, consider adding air purification features and integration with popular smart home ecosystems."

        The team can then review the conversation history, save this workflow for future sessions, and iterate on the ideas generated. They might run additional sessions with different prompts or agent combinations to explore various aspects of the product concept.

        This collaborative AI approach allows the team to rapidly generate and refine ideas, combining diverse perspectives and expertise in a structured, efficient manner.
        """
        )
        
