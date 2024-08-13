# nodes.py
import streamlit as st
import json
import os
import random
import logging
from typing import List, Dict, Union
from ollama_utils import get_available_models, call_ollama_endpoint, load_api_keys
from openai_utils import OPENAI_MODELS, call_openai_api
from groq_utils import GROQ_MODELS, call_groq_api
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define color scheme and emojis
NODE_COLORS = {
    "Input": "#90EE90",  # Light Green
    "LLM": "#ADD8E6",    # Light Blue
    "Output": "#FFB6C1"  # Light Pink
}

NODE_EMOJIS = {
    "Input": "📥",
    "LLM": "🧠",
    "Output": "📤"
}

class Node:
    def __init__(self, id: str, node_type: str, data: dict):
        self.id = id
        self.type = node_type
        self.data = data

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data
        }

    @staticmethod
    def from_dict(data: dict):
        return Node(data['id'], data['type'], data['data'])

class Edge:
    def __init__(self, id: str, source: str, target: str):
        self.id = id
        self.source = source
        self.target = target

    def to_dict(self):
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target
        }

    @staticmethod
    def from_dict(data: dict):
        return Edge(data['id'], data['source'], data['target'])

def execute_workflow(nodes: List[Node], edges: List[Edge]) -> Dict[str, str]:
    results = {}
    node_map = {node.id: node for node in nodes}
    api_keys = load_api_keys()

    def process_node(node_id):
        if node_id in results:
            return results[node_id]

        node = node_map[node_id]
        if node.type == 'Input':
            results[node_id] = node.data['input_text']
        elif node.type == 'LLM':
            incoming_edge = next((edge for edge in edges if edge.target == node_id), None)
            if incoming_edge:
                input_text = process_node(incoming_edge.source)
                prompt = node.data['prompt']
                complete_prompt = construct_prompt(node, prompt, input_text)
                
                if node.data['model_name'] in OPENAI_MODELS:
                    response = call_openai_api(
                        node.data['model_name'], 
                        [{"role": "user", "content": complete_prompt}],
                        temperature=node.data['temperature'],
                        max_tokens=node.data['max_tokens'],
                        openai_api_key=api_keys.get("openai_api_key")
                    )
                elif node.data['model_name'] in GROQ_MODELS:
                    response = call_groq_api(
                        node.data['model_name'],
                        complete_prompt,
                        temperature=node.data['temperature'],
                        max_tokens=node.data['max_tokens'],
                        groq_api_key=api_keys.get("groq_api_key")
                    )
                else:
                    response, _, _, _ = call_ollama_endpoint(
                        node.data['model_name'],
                        prompt=complete_prompt,
                        temperature=node.data['temperature'],
                        max_tokens=node.data['max_tokens'],
                        presence_penalty=node.data['presence_penalty'],
                        frequency_penalty=node.data['frequency_penalty']
                    )
                results[node_id] = response
            else:
                results[node_id] = "No input provided."
        elif node.type == 'Output':
            incoming_edge = next((edge for edge in edges if edge.target == node_id), None)
            if incoming_edge:
                input_text = process_node(incoming_edge.source)
                results[node_id] = f"{node.data['output_label']}: {input_text}"
            else:
                results[node_id] = "No input provided."
        
        return results[node_id]

    for node in nodes:
        process_node(node.id)

    return results

def construct_prompt(node: Node, base_prompt: str, input_text: str) -> str:
    prompt_parts = []
    if node.data.get('agent_type', 'None') != 'None':
        prompt_parts.append(get_agent_prompt()[node.data['agent_type']])
    if node.data.get('metacognitive_type', 'None') != 'None':
        prompt_parts.append(get_metacognitive_prompt()[node.data['metacognitive_type']])
    if node.data.get('voice_type', 'None') != 'None':
        prompt_parts.append(get_voice_prompt()[node.data['voice_type']])
    if node.data.get('identity_type', 'None') != 'None':
        prompt_parts.append(get_identity_prompt()[node.data['identity_type']])
    prompt_parts.append(base_prompt)
    prompt_parts.append(f"User Input: {input_text}")
    return "\n\n".join(prompt_parts)

def validate_workflow(nodes: List[Node], edges: List[Edge]) -> Dict[str, Union[bool, str]]:
    # Check if there's at least one Input node
    if not any(node.type == 'Input' for node in nodes):
        return {'valid': False, 'error': 'Workflow must have at least one Input node.'}

    # Check if there's at least one Output node
    if not any(node.type == 'Output' for node in nodes):
        return {'valid': False, 'error': 'Workflow must have at least one Output node.'}

    # Check if all nodes are connected
    node_ids = set(node.id for node in nodes)
    connected_nodes = set()
    for edge in edges:
        connected_nodes.add(edge.source)
        connected_nodes.add(edge.target)
    if node_ids != connected_nodes:
        return {'valid': False, 'error': 'All nodes must be connected in the workflow.'}

    # Check for cycles in the workflow
    if has_cycle(nodes, edges):
        return {'valid': False, 'error': 'Workflow contains a cycle. It must be acyclic.'}

    # Check if there's a path from an Input node to an Output node
    if not path_exists_input_to_output(nodes, edges):
        return {'valid': False, 'error': 'There must be a path from an Input node to an Output node.'}

    return {'valid': True, 'error': None}

def has_cycle(nodes: List[Node], edges: List[Edge]) -> bool:
    graph = {node.id: set() for node in nodes}
    for edge in edges:
        graph[edge.source].add(edge.target)

    visited = set()
    rec_stack = set()

    def is_cyclic(node_id):
        visited.add(node_id)
        rec_stack.add(node_id)

        for neighbor in graph[node_id]:
            if neighbor not in visited:
                if is_cyclic(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node_id)
        return False

    for node in nodes:
        if node.id not in visited:
            if is_cyclic(node.id):
                return True

    return False

def path_exists_input_to_output(nodes: List[Node], edges: List[Edge]) -> bool:
    graph = {node.id: set() for node in nodes}
    for edge in edges:
        graph[edge.source].add(edge.target)

    input_nodes = [node.id for node in nodes if node.type == 'Input']
    output_nodes = [node.id for node in nodes if node.type == 'Output']

    def dfs(node_id, visited):
        if node_id in output_nodes:
            return True
        visited.add(node_id)
        for neighbor in graph[node_id]:
            if neighbor not in visited:
                if dfs(neighbor, visited):
                    return True
        return False

    for input_node in input_nodes:
        if dfs(input_node, set()):
            return True

    return False

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define color scheme and emojis
NODE_COLORS = {
    "Input": "#90EE90",  # Light Green
    "LLM": "#ADD8E6",    # Light Blue
    "Output": "#FFB6C1"  # Light Pink
}

NODE_EMOJIS = {
    "Input": "📥",
    "LLM": "🧠",
    "Output": "📤"
}

def get_all_models():
    ollama_models = get_available_models()
    all_models = ollama_models + OPENAI_MODELS + GROQ_MODELS
    return all_models

class Node:
    def __init__(self, id: str, node_type: str, data: dict):
        self.id = id
        self.type = node_type
        self.data = data

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data
        }

    @staticmethod
    def from_dict(data: dict):
        return Node(data['id'], data['type'], data['data'])

class Edge:
    def __init__(self, id: str, source: str, target: str):
        self.id = id
        self.source = source
        self.target = target

    def to_dict(self):
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target
        }

    @staticmethod
    def from_dict(data: dict):
        return Edge(data['id'], data['source'], data['target'])

def create_node(node_id: str, node_type: str) -> Node:
    base_data = {
        'content': f'{node_type} Node',
    }
    if node_type == 'LLM':
        all_models = get_all_models()
        base_data.update({
            'model_name': all_models[0] if all_models else 'gpt-3.5-turbo',
            'agent_type': 'None',
            'metacognitive_type': 'None',
            'voice_type': 'None',
            'identity_type': 'None',
            'temperature': 0.7,
            'max_tokens': 4000,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'prompt': 'Enter Prompt:',
        })
    elif node_type == 'Input':
        base_data.update({
            'input_text': 'Enter Input:',
        })
    elif node_type == 'Output':
        base_data.update({
            'output_label': 'Output:',
            'document_format': 'Text'
        })
    return Node(node_id, node_type, base_data)

# ... [keep the execute_workflow, construct_prompt, save_workflow, and load_workflow functions as they are] ...

def render_node_settings(node: Node):
    st.sidebar.subheader(f"Configure {NODE_EMOJIS[node.type]} {node.type} Node {node.id}")
    node.data['content'] = st.sidebar.text_input("Node Label", value=node.data['content'], key=f"label_{node.id}")

    if node.type == 'LLM':
        all_models = get_all_models()
        node.data['model_name'] = st.sidebar.selectbox(
            "Select Model", 
            all_models, 
            index=all_models.index(node.data['model_name']) if node.data['model_name'] in all_models else 0, 
            key=f"model_{node.id}"
        )
        node.data['agent_type'] = st.sidebar.selectbox("Agent Type", ["None"] + list(get_agent_prompt().keys()), index=(["None"] + list(get_agent_prompt().keys())).index(node.data['agent_type']), key=f"agent_type_{node.id}")
        node.data['metacognitive_type'] = st.sidebar.selectbox("Metacognitive Type", ["None"] + list(get_metacognitive_prompt().keys()), index=(["None"] + list(get_metacognitive_prompt().keys())).index(node.data['metacognitive_type']), key=f"metacognitive_type_{node.id}")
        node.data['voice_type'] = st.sidebar.selectbox("Voice Type", ["None"] + list(get_voice_prompt().keys()), index=(["None"] + list(get_voice_prompt().keys())).index(node.data['voice_type']), key=f"voice_type_{node.id}")
        node.data['identity_type'] = st.sidebar.selectbox("Identity Type", ["None"] + list(get_identity_prompt().keys()), index=(["None"] + list(get_identity_prompt().keys())).index(node.data['identity_type']), key=f"identity_type_{node.id}")
        node.data['temperature'] = st.sidebar.slider("Temperature", 0.0, 1.0, node.data['temperature'], key=f"temperature_{node.id}")
        node.data['max_tokens'] = st.sidebar.slider("Max Tokens", 1000, 128000, node.data['max_tokens'], step=1000, key=f"max_tokens_{node.id}")
        node.data['presence_penalty'] = st.sidebar.slider("Presence Penalty", -2.0, 2.0, node.data['presence_penalty'], step=0.1, key=f"presence_penalty_{node.id}")
        node.data['frequency_penalty'] = st.sidebar.slider("Frequency Penalty", -2.0, 2.0, node.data['frequency_penalty'], step=0.1, key=f"frequency_penalty_{node.id}")
        node.data['prompt'] = st.sidebar.text_area("Prompt", value=node.data['prompt'], key=f"prompt_input_{node.id}")
    elif node.type == 'Input':
        node.data['input_text'] = st.sidebar.text_area("Input Text", value=node.data['input_text'], key=f"input_text_input_{node.id}")
    elif node.type == 'Output':
        node.data['output_label'] = st.sidebar.text_input("Output Label", value=node.data['output_label'], key=f"output_label_{node.id}")
        node.data['document_format'] = st.sidebar.selectbox("Document Format", ["Text", "Markdown", "HTML"], index=["Text", "Markdown", "HTML"].index(node.data['document_format']), key=f"document_format_{node.id}")

    if st.sidebar.button("Update Node", key=f"update_node_{node.id}"):
        st.success(f"Node {node.id} updated successfully!")
        st.rerun()

def render_workflow_canvas(nodes: List[Node], edges: List[Edge]):
    st.subheader("🔗 Workflow Canvas")
    
    # Create a grid layout for nodes
    cols = st.columns(3)
    for i, node in enumerate(nodes):
        with cols[i % 3]:
            with st.container(border=True):
                # Make the node clickable
                if st.button(f"{NODE_EMOJIS[node.type]} Node {node.id}: {node.type}", 
                             key=f"node_button_{node.id}",
                             use_container_width=True,
                             type="primary"):
                    st.session_state.selected_node_id = node.id
                    st.rerun()
                
                st.markdown(f"<div style='background-color: {NODE_COLORS[node.type]}; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
                st.write(f"**Content:** {node.data['content']}")
                if node.type == 'LLM':
                    st.write(f"**Model:** {node.data['model_name']}")
                elif node.type == 'Input':
                    st.write(f"**Input:** {node.data['input_text'][:50]}...")
                elif node.type == 'Output':
                    st.write(f"**Format:** {node.data['document_format']}")
                st.markdown("</div>", unsafe_allow_html=True)

    # Render edges
    st.subheader("🔀 Connections")
    for edge in edges:
        st.markdown(f"<div style='text-align: center; font-weight: bold;'>Node {edge.source} ➡️ Node {edge.target}</div>", unsafe_allow_html=True)

def nodes_interface():
    st.title("🧩 LLM Workflow Builder")

    # Add custom CSS for danger buttons
    st.markdown("""
    <style>
    div.stButton > button:first-child.stBtn.secondary {
        background-color: #dc3545;
        color: white;
    }
    div.stButton > button:hover:first-child.stBtn.secondary {
        background-color: #bd2130;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if 'nodes' not in st.session_state:
        st.session_state['nodes'] = [
            create_node("1", "Input"),
            create_node("2", "LLM"),
            create_node("3", "Output")
        ]
    if 'edges' not in st.session_state:
        st.session_state['edges'] = [
            Edge("1-2", "1", "2"),
            Edge("2-3", "2", "3")
        ]
    if 'selected_node_id' not in st.session_state:
        st.session_state.selected_node_id = None

    with st.sidebar:
        st.subheader("🛠️ Workflow Actions")
        
        st.subheader("➕ Add New Node")
        node_type = st.selectbox("Select node type", ["Input", "LLM", "Output"])
        if st.button("Add Node", type="primary"):
            new_node_id = str(len(st.session_state['nodes']) + 1)
            new_node = create_node(new_node_id, node_type)
            st.session_state['nodes'].append(new_node)
            st.rerun()

        st.subheader("➕ Add New Edge")
        source = st.selectbox("Source Node", [node.id for node in st.session_state['nodes']], key="edge_source")
        target = st.selectbox("Target Node", [node.id for node in st.session_state['nodes']], key="edge_target")
        if st.button("Add Edge", type="primary"):
            new_edge = Edge(f"{source}-{target}", source, target)
            if new_edge not in st.session_state['edges']:
                st.session_state['edges'].append(new_edge)
                st.rerun()

        st.subheader("⚙️ Configure Node")
        selected_node = next((node for node in st.session_state['nodes'] if node.id == st.session_state.selected_node_id), None)
        if selected_node:
            render_node_settings(selected_node)
        else:
            st.info("Select a node from the canvas to configure it.")

    render_workflow_canvas(st.session_state['nodes'], st.session_state['edges'])

    st.subheader("🎛️ Workflow Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Execute Workflow", type="primary"):
            with st.spinner("Executing workflow..."):
                results = execute_workflow(st.session_state['nodes'], st.session_state['edges'])
            st.write("Workflow Execution Results:")
            for node_id, result in results.items():
                node = next(node for node in st.session_state['nodes'] if node.id == node_id)
                st.subheader(f"{NODE_EMOJIS[node.type]} Node {node_id} ({node.type}):")
                if node.type == 'Output':
                    if node.data['document_format'] == 'Markdown':
                        st.markdown(result)
                    elif node.data['document_format'] == 'HTML':
                        st.components.v1.html(result, height=300)
                    else:
                        st.text(result)
                else:
                    st.text(result)
    with col2:
        if st.button("💾 Save Workflow", type="secondary"):
            save_workflow(st.session_state['nodes'], st.session_state['edges'])
            st.success("Workflow saved successfully!")
    with col3:
        if st.button("📂 Load Workflow", type="secondary"):
            loaded_nodes, loaded_edges = load_workflow()
            if loaded_nodes:
                st.session_state['nodes'] = loaded_nodes
                st.session_state['edges'] = loaded_edges
                st.success("Workflow loaded successfully!")
                st.rerun()
            else:
                st.warning("No saved workflow found.")

    # Display current workflow information
    st.subheader("📊 Current Workflow Information")
    st.write(f"Number of Nodes: {len(st.session_state['nodes'])}")
    st.write(f"Number of Edges: {len(st.session_state['edges'])}")
    
    # Display node information
    st.subheader("🔍 Node Information")
    for node in st.session_state['nodes']:
        with st.expander(f"{NODE_EMOJIS[node.type]} Node {node.id} ({node.type})"):
            st.write(f"**Content:** {node.data['content']}")
            st.write(f"**Type:** {node.type}")
            if node.type == 'LLM':
                st.write(f"**Model:** {node.data['model_name']}")
                st.write(f"**Agent Type:** {node.data['agent_type']}")
                st.write(f"**Metacognitive Type:** {node.data['metacognitive_type']}")
                st.write(f"**Voice Type:** {node.data['voice_type']}")
                st.write(f"**Identity Type:** {node.data['identity_type']}")
                st.write(f"**Temperature:** {node.data['temperature']}")
                st.write(f"**Max Tokens:** {node.data['max_tokens']}")
                st.write(f"**Presence Penalty:** {node.data['presence_penalty']}")
                st.write(f"**Frequency Penalty:** {node.data['frequency_penalty']}")
                st.text_area("Prompt", value=node.data['prompt'], disabled=True, key=f"prompt_display_{node.id}")
            elif node.type == 'Input':
                st.text_area("Input Text", value=node.data['input_text'], disabled=True, key=f"input_text_display_{node.id}")
            elif node.type == 'Output':
                st.write(f"**Output Label:** {node.data['output_label']}")
                st.write(f"**Document Format:** {node.data['document_format']}")

    # Display edge information
    st.subheader("🔗 Edge Information")
    for edge in st.session_state['edges']:
        st.write(f"Edge: Node {edge.source} ➡️ Node {edge.target}")

    # Add option to remove nodes and edges
    st.subheader("🗑️ Remove Nodes or Edges")
    remove_type = st.radio("Select what to remove:", ["Node", "Edge"])
    if remove_type == "Node":
        node_to_remove = st.selectbox("Select node to remove:", [f"{NODE_EMOJIS[node.type]} Node {node.id}" for node in st.session_state['nodes']])
        if st.button("Remove Node", type="secondary", key="remove_node_button"):
            node_id = node_to_remove.split()[-1]
            st.session_state['nodes'] = [node for node in st.session_state['nodes'] if node.id != node_id]
            st.session_state['edges'] = [edge for edge in st.session_state['edges'] if edge.source != node_id and edge.target != node_id]
            st.success(f"Node {node_id} and its connected edges removed.")
            st.rerun()
    else:
        edge_to_remove = st.selectbox("Select edge to remove:", [f"Node {edge.source} ➡️ Node {edge.target}" for edge in st.session_state['edges']])
        if st.button("Remove Edge", type="secondary", key="remove_edge_button"):
            source, target = edge_to_remove.split("➡️")
            source = source.split()[-1]
            target = target.split()[-1]
            st.session_state['edges'] = [edge for edge in st.session_state['edges'] if not (edge.source == source and edge.target == target)]
            st.success(f"Edge {edge_to_remove} removed.")
            st.rerun()

    # Add a section for workflow validation
    st.subheader("🔍 Workflow Validation")
    if st.button("Validate Workflow", type="primary"):
        validation_result = validate_workflow(st.session_state['nodes'], st.session_state['edges'])
        if validation_result['valid']:
            st.success("Workflow is valid! ✅")
        else:
            st.error(f"Workflow validation failed: {validation_result['error']}")

    # Add a section for workflow statistics
    st.subheader("📊 Workflow Statistics")
    st.write(f"Total Nodes: {len(st.session_state['nodes'])}")
    st.write(f"Total Edges: {len(st.session_state['edges'])}")

    # Count node types
    node_types = {}
    for node in st.session_state['nodes']:
        node_types[node.type] = node_types.get(node.type, 0) + 1

    st.write("Node Types:")
    for node_type, count in node_types.items():
        st.write(f"  - {NODE_EMOJIS[node_type]} {node_type}: {count}")

    # Add a help section
    with st.expander("❓ Help"):
        st.markdown("""
        ### How to use the LLM Workflow Builder:
        1. **Add Nodes**: Use the sidebar to add Input, LLM, and Output nodes.
        2. **Connect Nodes**: Add edges to connect nodes in the desired order.
        3. **Configure Nodes**: Click on a node in the canvas to configure its settings in the sidebar.
        4. **Execute Workflow**: Click the 'Execute Workflow' button to run your workflow.
        5. **Save/Load**: Save your workflow for later use or load a previously saved workflow.
        6. **Validate**: Use the 'Validate Workflow' button to check if your workflow is properly constructed.
        7. **Remove**: Remove nodes or edges using the removal section at the bottom.

        For more detailed instructions, please refer to the documentation.
        """)

if __name__ == "__main__":
    nodes_interface()