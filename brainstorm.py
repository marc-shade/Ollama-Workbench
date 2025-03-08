# brainstorm.py
import os
import openai
import json
import streamlit as st
import subprocess
try:
    from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    print("Warning: autogen package not found, using fallback implementation")
    # Fallback implementation for autogen
    class ConversableAgent:
        def __init__(self, name, llm_config=None, human_input_mode=None, **kwargs):
            self.name = name
            self.llm_config = llm_config
            print(f"Warning: Using fallback ConversableAgent with name: {name}")
            
        def generate_reply(self, messages, sender, config=None):
            print(f"Warning: Using fallback generate_reply for {self.name}")
            return f"This is a fallback response from {self.name}. The autogen package is not installed."
    
    class UserProxyAgent(ConversableAgent):
        def __init__(self, name, human_input_mode=None, code_execution_config=None, **kwargs):
            super().__init__(name=name, **kwargs)
            self.human_input_mode = human_input_mode
            self.code_execution_config = code_execution_config
    
    class GroupChat:
        def __init__(self, agents, messages=None, speaker_selection_method=None):
            self.agents = agents
            self.messages = messages or []
            self.speaker_selection_method = speaker_selection_method
    
    class GroupChatManager:
        def __init__(self, groupchat):
            self.groupchat = groupchat
try:
    from autogen.agentchat.contrib.capabilities.teachability import Teachability
except ImportError:
    print("Warning: autogen.agentchat.contrib.capabilities.teachability package not found, using fallback implementation")
    # Fallback implementation for Teachability
    class Teachability:
        def __init__(self, path_to_db_dir=None):
            self.path_to_db_dir = path_to_db_dir
            print(f"Warning: Using fallback Teachability with path: {path_to_db_dir}")
            
        def add_to_agent(self, agent):
            print(f"Warning: Using fallback add_to_agent for {agent.name}")
            pass
from ollama_utils import get_available_models, get_all_models
import markdown
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt
from info_brainstorm import display_info_brainstorm
from chat_interface import chat_interface
from ollama_utils import load_api_keys
from groq_utils import GROQ_MODELS
from openai_utils import OPENAI_MODELS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SETTINGS_FILE = "brainstorm_agents_settings.json"
WORKFLOWS_DIR = "brainstorm_workflows"
API_KEYS_FILE = "api_keys.json"

# List of animal emojis
ANIMAL_EMOJIS = [
    "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯", "🦁", "🐮", "🐷", "🐸", "🐵", "🐔", "🐧", "🐦", "🐤",
    "🦆", "🦅", "🦉", "🦇", "🐺", "🐗", "🐴", "🦄", "🐝", "🐛", "🦋", "🐌", "🐞", "🐜", "🦟", "🦗", "🕷", "🦂", "🐢",
    "🐍", "🦎", "🦖", "🦕", "🐙", "🦑", "🦐", "🦞", "🦀", "🐡", "🐠", "🐟", "🐬", "🐳", "🐋", "🦈", "🐊", "🐅", "🐆",
    "🦓", "🦍", "🦧", "🐘", "🦛", "🦏", "🐪", "🐫", "🦒", "🦘", "🐃", "🐂", "🐄", "🐎", "🐖", "🐏", "🐑", "🦙", "🐐",
    "🦌", "🐕", "🐩", "🦮", "🐕‍🦺", "🐈", "🐈‍⬛", "🐓", "🦃", "🦚", "🦜", "🦢", "🦩", "🕊", "🐇", "🦝", "🦨", "🦡", "🦦",
    "🦥", "🐁", "🐀", "🐿", "🦔", "🐾", "🐉", "🐲", "🤖", "🧚"
]

try:
    from files_management import get_corpus_options
except ImportError:
    def get_corpus_options(): return []

class CustomConversableAgent(ConversableAgent):
    def __init__(self, name, llm_config, agent_type, identity, metacognitive_type, voice_type, corpus, temperature, max_tokens, presence_penalty, frequency_penalty, db_path, *args, **kwargs):
        if 'api_key' in llm_config:
            os.environ["OPENAI_API_KEY"] = llm_config['api_key']
        
        super().__init__(name=name, llm_config=llm_config, *args, **kwargs)
        self.agent_type = agent_type
        self.identity = identity
        self.metacognitive_type = metacognitive_type
        self.voice_type = voice_type
        self.corpus = corpus
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.teachability = Teachability(path_to_db_dir=db_path)
        self.teachability.add_to_agent(self)

    def generate_reply(self, messages, sender, config):
        context = "\n".join([f"{m['role']} ({m.get('name', 'Unknown')}): {m['content']}" for m in messages])
        prompt = f"""
        You are a {self.agent_type} agent named {self.name}.
        Identity: {self.identity}
        Metacognitive Type: {self.metacognitive_type}
        Voice Type: {self.voice_type}
        Corpus: {self.corpus}

        Full conversation context:
        {context}

        Your task is to respond to the latest message, considering the full context above. Remember your role and respond accordingly:
        """
        return super().generate_reply(messages=[{"role": "user", "content": prompt}], sender=sender, config=config)

def brainstorm_session():
    st.header("💡 Brainstorming Session")
    
    st.write("Welcome to the brainstorming session! Here you can discuss ideas, share thoughts, and collaborate in real-time.")
    
    chat_interface()

def create_agent(settings):
    api_keys = load_api_keys()  # Load API keys
    print(f"Creating agent with model: {settings['model']}")
    print(f"API keys loaded: {api_keys}")
    
    if settings['model'] in OPENAI_MODELS:
        print("Using OpenAI model")
        llm_config = {
            "request_timeout": 120,
            "api_key": api_keys.get("openai_api_key"),
            "model": settings['model'],
        }
        # Set the OpenAI API key in the environment variable
        os.environ["OPENAI_API_KEY"] = api_keys.get("openai_api_key", "")
    elif settings['model'] in GROQ_MODELS:
        print("Using Groq model")
        llm_config = {
            "request_timeout": 120,
            "api_key": api_keys.get("groq_api_key"),
            "model": settings['model'],
        }
    else:
        print("Using Ollama model")
        llm_config = {
            "request_timeout": 120,
            "api_base": "http://localhost:11434/v1",
            "api_type": "open_ai",
            "model": settings["model"],
        }
    
    print(f"LLM config: {llm_config}")
    
    return CustomConversableAgent(
        name=f"{settings['emoji']} {settings['name']}",
        llm_config=llm_config,
        agent_type=settings["agent_type"],
        identity=settings["identity"],
        metacognitive_type=settings["metacognitive_type"],
        voice_type=settings["voice_type"],
        corpus=settings["corpus"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        presence_penalty=settings["presence_penalty"],
        frequency_penalty=settings["frequency_penalty"],
        db_path=settings["db_path"]
    )
    
def load_agent_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {"agents": []}

def save_agent_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def get_available_workflows():
    if not os.path.exists(WORKFLOWS_DIR):
        os.makedirs(WORKFLOWS_DIR)
    return [f[:-5] for f in os.listdir(WORKFLOWS_DIR) if f.endswith('.json')]

def save_workflow(workflow_name, agent_sequence):
    if not os.path.exists(WORKFLOWS_DIR):
        os.makedirs(WORKFLOWS_DIR)
    workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_name}.json")
    with open(workflow_path, 'w') as f:
        json.dump(agent_sequence, f, indent=2)
    st.success(f"Workflow '{workflow_name}' saved successfully!")
    print(f"Saved workflow: {workflow_name}")
    print(f"Agent sequence: {agent_sequence}")

def load_workflow(workflow_name):
    workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_name}.json")
    if os.path.exists(workflow_path):
        with open(workflow_path, 'r') as f:
            agent_sequence = json.load(f)
        print(f"Loaded workflow: {workflow_name}")
        print(f"Agent sequence: {agent_sequence}")
        return agent_sequence
    return None

def edit_agent_settings(agent_settings):
    st.subheader(f"Edit Agent: {agent_settings['name']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        agent_settings['name'] = st.text_input("Agent Name", agent_settings['name'], key=f"{agent_settings['name']}_name")
        agent_settings['emoji'] = st.selectbox("Emoji", ANIMAL_EMOJIS, index=ANIMAL_EMOJIS.index(agent_settings.get('emoji', '🐶')), key=f"{agent_settings['name']}_emoji")
        agent_settings['model'] = st.selectbox("Model", get_all_models(), 
                                               index=get_all_models().index(agent_settings['model']) if agent_settings['model'] in get_all_models() else 0,
                                               key=f"{agent_settings['name']}_model")
        voice_options = ["None"] + list(get_voice_prompt().keys())
        agent_settings['voice_type'] = st.selectbox(
            "Voice Type",
            voice_options,
            index=voice_options.index(agent_settings['voice_type']) if agent_settings['voice_type'] in voice_options else 0,
            key=f"{agent_settings['name']}_voice_type"
        )
    
    with col2:
        agent_type_options = ["None"] + list(get_agent_prompt().keys())
        agent_settings['agent_type'] = st.selectbox(
            "Agent Type",
            agent_type_options,
            index=agent_type_options.index(agent_settings['agent_type']) if agent_settings['agent_type'] in agent_type_options else 0,
            key=f"{agent_settings['name']}_agent_type"
        )
        identity_options = ["None"] + list(get_identity_prompt().keys())
        agent_settings['identity'] = st.selectbox(
            "Identity",
            identity_options,
            index=identity_options.index(agent_settings['identity']) if agent_settings['identity'] in identity_options else 0,
            key=f"{agent_settings['name']}_identity"
        )
        metacognitive_options = ["None"] + list(get_metacognitive_prompt().keys())
        agent_settings['metacognitive_type'] = st.selectbox(
            "Metacognitive Type",
            metacognitive_options,
            index=metacognitive_options.index(agent_settings['metacognitive_type']) if agent_settings['metacognitive_type'] in metacognitive_options else 0,
            key=f"{agent_settings['name']}_metacognitive_type"
        )
        corpus_options = ["None"] + get_corpus_options()
        agent_settings['corpus'] = st.selectbox(
            "Corpus",
            corpus_options,
            index=corpus_options.index(agent_settings['corpus']) if agent_settings['corpus'] in corpus_options else 0,
            key=f"{agent_settings['name']}_corpus"
        )
    
    with col3:
        agent_settings['temperature'] = st.slider("Temperature", 0.0, 1.0, agent_settings['temperature'], key=f"{agent_settings['name']}_temperature")
        agent_settings['max_tokens'] = st.slider("Max Tokens", min_value=1000, max_value=128000, value=agent_settings['max_tokens'], step=1000, key=f"{agent_settings['name']}_max_tokens")
        agent_settings['presence_penalty'] = st.slider("Presence Penalty", -2.0, 2.0, agent_settings['presence_penalty'], key=f"{agent_settings['name']}_presence_penalty")
        agent_settings['frequency_penalty'] = st.slider("Frequency Penalty", -2.0, 2.0, agent_settings['frequency_penalty'], key=f"{agent_settings['name']}_frequency_penalty")
        agent_settings['db_path'] = st.text_input("Database Path", agent_settings['db_path'], key=f"{agent_settings['name']}_db_path")

    return agent_settings

def manage_agents():
    st.subheader("Manage Agents")
    settings = load_agent_settings()
    
    for i, agent in enumerate(settings["agents"]):
        with st.expander(f"{agent.get('emoji', '🐶')} {agent['name']}"):
            updated_agent = edit_agent_settings(agent)
            settings["agents"][i] = updated_agent
            if st.button(f"Remove {agent['name']}", key=f"remove_{agent['name']}"):
                settings["agents"].pop(i)
                save_agent_settings(settings)
                st.success(f"Agent {agent['name']} removed.")
                st.rerun()
    
    with st.expander("➕ Add New Agent"):
        new_agent = {
            "name": "",
            "emoji": "🐶",
            "model": get_all_models()[0],
            "agent_type": "None",
            "identity": "None",
            "metacognitive_type": "None",
            "voice_type": "None",
            "corpus": "None",
            "temperature": 0.7,
            "max_tokens": 4000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "db_path": "./tmp/new_agent_db"
        }
        
        new_agent = edit_agent_settings(new_agent)
        
        if st.button("Add Agent"):
            if new_agent["name"]:
                new_agent['db_path'] = f"./tmp/{new_agent['name'].lower().replace(' ', '_')}_db"
                settings["agents"].append(new_agent)
                save_agent_settings(settings)
                st.success(f"Agent {new_agent['name']} added successfully!")
                st.rerun()
            else:
                st.error("Please provide a name for the new agent.")
    
    save_agent_settings(settings)

def brainstorm_session(use_docker):
    settings = load_agent_settings()
    api_keys = load_api_keys()
    os.environ["OPENAI_API_KEY"] = api_keys.get("openai_api_key", "")
    openai.api_key = api_keys.get("openai_api_key", "")
    print(f"OpenAI API Key: {os.environ.get('OPENAI_API_KEY')}")
    agents = [create_agent(agent_settings) for agent_settings in settings["agents"]]

    if 'group_chat' not in st.session_state:
        user = UserProxyAgent(
            "👨‍💼 User",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": use_docker}
        )
        st.session_state.group_chat = GroupChat(
            agents=[*agents, user],
            messages=[],
            speaker_selection_method="manual"
        )
        st.session_state.group_chat_manager = GroupChatManager(groupchat=st.session_state.group_chat)

    # Workflow management
    available_workflows = get_available_workflows()
    col1, col2, col3 = st.columns([10, 1, 10], vertical_alignment="bottom")
    with col1:
        workflow_name = st.text_input("Add a New Workflow")
    with col2:
        st.write("OR")
    with col3:
        selected_workflow = st.selectbox("Load an Existing Workflow", [""] + available_workflows)

    if selected_workflow and selected_workflow != st.session_state.get('last_loaded_workflow'):
        agent_sequence = load_workflow(selected_workflow)
        if agent_sequence:
            st.session_state.agent_sequence = agent_sequence
            st.session_state.last_loaded_workflow = selected_workflow
            st.success(f"Workflow '{selected_workflow}' loaded successfully!")
            st.rerun()

    # Agent sequence setup
    st.subheader("🦊 Agent Response Sequence")
    if 'agent_sequence' not in st.session_state:
        st.session_state.agent_sequence = []

    agent_names = [f"{agent.name}" for agent in agents]
    agent_names.insert(0, "")  # Add empty option at the beginning

    # Number of agents in the workflow
    num_agents = len(st.session_state.agent_sequence)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.write("Number of Agents:")
    with col2:
        num_agents = st.number_input(
            "Number of Agents in Workflow",
            min_value=0,
            value=num_agents,
            step=1,
            label_visibility="collapsed"
        )

    # Display agent selection dropdowns
    for i in range(num_agents):
        current_agent = st.session_state.agent_sequence[i] if i < len(st.session_state.agent_sequence) else ""
        agent = st.selectbox(
            f"Agent {i+1}",
            agent_names,
            index=agent_names.index(current_agent) if current_agent in agent_names else 0,
            key=f"agent_{i}"
        )
        if i < len(st.session_state.agent_sequence):
            st.session_state.agent_sequence[i] = agent
        else:
            st.session_state.agent_sequence.append(agent)

    if st.button("💾 Save Workflow"):
        if workflow_name:
            current_sequence = [agent for agent in st.session_state.agent_sequence if agent]
            save_workflow(workflow_name, current_sequence)
        else:
            st.error("Please enter a workflow name before saving.")

    # User input
    st.subheader("🧑‍⚕️ User Input")
    user_message = st.text_input("Enter your question or topic for the Brainstorm session:")

    if st.button("👥 Send Input to Agents"):
        if user_message and any(st.session_state.agent_sequence):
            # Add user message to the group chat
            st.session_state.group_chat.messages.append({"role": "user", "name": "User", "content": user_message})

            # Generate responses from the sequence of agents
            for agent_name in st.session_state.agent_sequence:
                if agent_name:  # Skip empty selections
                    with st.spinner(f"{agent_name} is thinking..."):
                        selected_agent_obj = next((agent for agent in agents if agent.name == agent_name), None)
                        if selected_agent_obj:
                            response = selected_agent_obj.generate_reply(st.session_state.group_chat.messages, sender=st.session_state.group_chat_manager, config=None)
                            # Add agent's response to the group chat
                            st.session_state.group_chat.messages.append({"role": "assistant", "name": agent_name, "content": response})
                        else:
                            st.error(f"Agent '{agent_name}' not found.")

    # Display conversation history with formatting
    st.subheader("📜 Conversation History")
    for message in st.session_state.group_chat.messages:
        with st.chat_message(message['role']):
            st.markdown(f"**{message['name']}**")
            formatted_content = markdown.markdown(message['content'])
            st.markdown(formatted_content, unsafe_allow_html=True)

    # Display info about Brainstorm feature
    display_info_brainstorm()

def brainstorm_interface():
    st.title("🧠 Brainstorm")

    # Docker usage option
    use_docker = st.checkbox("Use Docker for code execution", value=False)

    if use_docker:
        # Check if Docker is installed and running
        if not is_docker_running():
            st.warning("🟠 Docker is not running. Please start Docker or install it if not already installed.")
            st.info("""
            ❓ To install Docker:
            1. Visit https://www.docker.com/products/docker-desktop
            2. Download and install Docker Desktop for your operating system
            3. After installation, start Docker Desktop
            4. Once Docker is running, refresh this page
            """)
            st.stop()
    else:
        os.environ["AUTOGEN_USE_DOCKER"] = "0"

    tab1, tab2 = st.tabs(["💡 Brainstorm Session", "🦊 Manage Agents"])

    with tab1:
        brainstorm_session(use_docker)

    with tab2:
        manage_agents()

def is_docker_running():
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

if __name__ == "__main__":
    brainstorm_interface()