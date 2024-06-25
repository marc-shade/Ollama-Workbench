import streamlit as st
import os
from ollama_utils import get_available_models, ollama
from prompts import get_agent_prompt, get_metacognitive_prompt

def get_corpus_options():
    corpus_folder = "corpus"
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder)
    return ["None"] + [f for f in os.listdir(corpus_folder) if os.path.isdir(os.path.join(corpus_folder, f))]

def get_corpus_context_from_db(corpus_folder, corpus, user_input):
    # Placeholder for the actual implementation of retrieving context from the corpus database
    return "Sample context from the corpus database."

def ai_agent(user_input, model, agent_type, metacognitive_type, corpus, temperature, max_tokens):
    # Combine agent type and metacognitive type prompts
    combined_prompt = ""
    if agent_type != "None":
        combined_prompt += get_agent_prompt()[agent_type] + "\n\n"
    if metacognitive_type != "None":
        combined_prompt += get_metacognitive_prompt()[metacognitive_type] + "\n\n"

    # Include corpus context if selected
    if corpus != "None":
        corpus_context = get_corpus_context_from_db("corpus", corpus, user_input)
        final_prompt = f"{combined_prompt}Context: {corpus_context}\n\nUser: {user_input}"
    else:
        final_prompt = f"{combined_prompt}User: {user_input}"

    # Generate response using Ollama
    response = ollama.generate(model, final_prompt, temperature=temperature, max_tokens=max_tokens)
    
    return response['response']

def define_agent_block(name):
    st.header(f"{name} Parameters")
    model = st.selectbox(f"{name} Model", get_available_models(), key=f"{name}_model")
    agent_type = st.selectbox(f"{name} Agent Type", ["None"] + list(get_agent_prompt().keys()), key=f"{name}_agent_type")
    metacognitive_type = st.selectbox(f"{name} Metacognitive Type", ["None"] + list(get_metacognitive_prompt().keys()), key=f"{name}_metacognitive_type")
    corpus = st.selectbox(f"{name} Corpus", get_corpus_options(), key=f"{name}_corpus")
    temperature = st.slider(f"{name} Temperature", 0.0, 1.0, 0.7, key=f"{name}_temperature")
    max_tokens = st.slider(f"{name} Max Tokens", 100, 32000, 4000, key=f"{name}_max_tokens")
    return model, agent_type, metacognitive_type, corpus, temperature, max_tokens

def main():
    st.title("AI Workflow Builder")

    st.header("Define Workflow")
    num_agents = st.number_input("Number of Agents", min_value=1, max_value=10, value=1)

    agents = []
    for i in range(num_agents):
        with st.expander(f"Agent {i+1}"):
            agents.append(define_agent_block(f"Agent {i+1}"))

    if st.button("Run Workflow"):
        st.header("Workflow Results")
        user_input = st.text_input("Initial User Input")
        current_input = user_input

        for i, (model, agent_type, metacognitive_type, corpus, temperature, max_tokens) in enumerate(agents):
            with st.spinner(f"Running Agent {i+1}..."):
                current_input = ai_agent(current_input, model, agent_type, metacognitive_type, corpus, temperature, max_tokens)
                st.write(f"Agent {i+1} Output: {current_input}")

if __name__ == "__main__":
    main()
