# nodes.py
import streamlit as st

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout, RadialLayout, ForceLayout, StressLayout, RandomLayout

from ollama_utils import get_available_models, call_ollama_endpoint, load_api_keys
from openai_utils import OPENAI_MODELS
from groq_utils import GROQ_MODELS
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt
import json
import os
import re
import random
import logging
from typing import List, Dict

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_models():
    ollama_models = get_available_models()
    all_models = ollama_models + OPENAI_MODELS + GROQ_MODELS
    return all_models

class StreamlitFlowNode:
    def __init__(self, id: str, position: tuple, data: dict, type: str = 'default', source_position: str = 'right', target_position: str = 'left', connectable: bool = True):
        self.id = id
        self.position = {'x': position[0], 'y': position[1]}  # Ensure position is a dictionary with x and y
        self.data = data
        self.type = type
        self.source_position = source_position
        self.target_position = target_position
        self.connectable = connectable

    def __to_dict__(self):
        return {
            'id': self.id,
            'position': self.position,
            'data': self.data,
            'type': self.type,
            'source_position': self.source_position,
            'target_position': self.target_position,
            'connectable': self.connectable
        }

    @staticmethod
    def from_serializable(data: dict):
        return StreamlitFlowNode(
            id=data['id'],
            position=(data['position']['x'], data['position']['y']),  # Convert position back to tuple
            data=data['data'],
            type=data['type'],
            source_position=data['source_position'],
            target_position=data['target_position'],
            connectable=data['connectable']
        )


class StreamlitFlowEdge:
    def __init__(self, id: str, source: str, target: str, animated: bool = True):
        self.id = id
        self.source = source
        self.target = target
        self.animated = animated

    def __to_dict__(self):
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'animated': self.animated
        }

    @staticmethod
    def from_serializable(data: dict):
        return StreamlitFlowEdge(
            id=data['id'],
            source=data['source'],
            target=data['target'],
            animated=data['animated']
        )


def create_node(node_id: str, node_type: str, position: tuple) -> StreamlitFlowNode:
    """Creates a new node with default settings based on its type."""
    base_data = {
        'id': node_id,
        'type': node_type,
        'content': f'{node_type} Node',
        'position': position,  # Add position to the node's data
    }
    if node_type == 'LLM':
        all_models = get_all_models()
        base_data.update({
            'model_name': all_models[0] if all_models else 'gpt-3.5-turbo',  # Use the first available model or a default
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

    source_position = 'right'
    target_position = 'left'

    if node_type == 'Input':
        source_position = 'right'
        target_position = 'left'
    elif node_type == 'Output':
        source_position = 'left'
        target_position = 'right'

    return StreamlitFlowNode(
        id=node_id,
        position=position,
        data=base_data,
        type='default' if node_type == 'LLM' else ('input' if node_type == 'Input' else 'output'),
        source_position=source_position,
        target_position=target_position,
        connectable=True
    )

def execute_workflow(nodes: List[StreamlitFlowNode], edges: List[StreamlitFlowEdge]) -> Dict[str, str]:
    """Executes the workflow based on the nodes and edges."""
    results = {}
    node_map = {node.id: node for node in nodes}
    api_keys = load_api_keys()

    def process_node(node_id):
        """Recursively processes a node and its inputs."""
        if node_id in results:
            return results[node_id]

        node = node_map[node_id]
        if node.data['type'] == 'Input':
            results[node_id] = node.data['input_text']
        elif node.data['type'] == 'LLM':
            incoming_edge = next((edge for edge in edges if edge.target == node_id), None)
            if incoming_edge:
                input_text = process_node(incoming_edge.source)
                prompt = node.data['prompt']
                complete_prompt = construct_prompt(node, prompt, input_text)
                
                if node.data['model_name'] in OPENAI_MODELS:
                    from openai_utils import call_openai_api
                    response = call_openai_api(
                        node.data['model_name'], 
                        [{"role": "user", "content": complete_prompt}],
                        temperature=node.data['temperature'],
                        max_tokens=node.data['max_tokens'],
                        openai_api_key=api_keys.get("openai_api_key")
                    )
                elif node.data['model_name'] in GROQ_MODELS:
                    from groq_utils import call_groq_api
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
        elif node.data['type'] == 'Output':
            incoming_edge = next((edge for edge in edges if edge.target == node_id), None)
            if incoming_edge:
                input_text = process_node(incoming_edge.source)
                results[node_id] = f"{node.data['output_label']}: {input_text}"
            else:
                results[node_id] = "No input provided."
        
        return results[node_id]

    # Process all nodes
    for node in nodes:
        process_node(node.id)

    return results

def construct_prompt(node: StreamlitFlowNode, base_prompt: str, input_text: str) -> str:
    """Constructs the complete prompt for an LLM node."""
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

def save_workflow(nodes: List[StreamlitFlowNode], edges: List[StreamlitFlowEdge], filename: str = "workflow.json"):
    """Saves the current workflow to a JSON file."""
    workflow_data = {
        "nodes": [node.__to_dict__() for node in nodes],
        "edges": [edge.__to_dict__() for edge in edges]
    }
    with open(filename, "w") as f:
        json.dump(workflow_data, f, indent=4)

def load_workflow(filename: str = "workflow.json") -> tuple:
    """Loads a workflow from a JSON file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            workflow_data = json.load(f)
        nodes = [StreamlitFlowNode.from_serializable(node_data) for node_data in workflow_data["nodes"]]
        edges = [StreamlitFlowEdge.from_serializable(edge_data) for edge_data in workflow_data["edges"]]
        return nodes, edges
    else:
        return [], []

def render_node_settings(node: StreamlitFlowNode):
    """Renders the settings panel for the selected node."""
    st.sidebar.subheader(f"Configure {node.data['type']} Node {node.id}")
    node.data['content'] = st.sidebar.text_input("Node Label", value=node.data['content'], key=f"label_{node.id}")

    if node.data['type'] == 'LLM':
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
        node.data['prompt'] = st.sidebar.text_area("Prompt", value=node.data['prompt'], key=f"prompt_{node.id}")
    elif node.data['type'] == 'Input':
        node.data['input_text'] = st.sidebar.text_area("Input Text", value=node.data['input_text'], key=f"input_text_{node.id}")
    elif node.data['type'] == 'Output':
        node.data['output_label'] = st.sidebar.text_input("Output Label", value=node.data['output_label'], key=f"output_label_{node.id}")
        node.data['document_format'] = st.sidebar.selectbox("Document Format", ["Text", "Markdown", "HTML"], index=["Text", "Markdown", "HTML"].index(node.data['document_format']), key=f"document_format_{node.id}")

    if st.sidebar.button("Update Node", key=f"update_node_{node.id}"):
        for i, n in enumerate(st.session_state['nodes']):
            if n.id == node.id:
                st.session_state['nodes'][i] = node
                break
        st.success(f"Node {node.id} updated successfully!")
        st.rerun()

def nodes_interface():
    st.title("LLM Workflow Builder")
    api_keys = load_api_keys()
    
    if 'nodes' not in st.session_state:
        st.session_state['nodes'] = [
            create_node("1", "Input", (0, 0)),
            create_node("2", "LLM", (300, 0)),
            create_node("3", "Output", (600, 0))
        ]
    if 'edges' not in st.session_state:
        st.session_state['edges'] = [
            StreamlitFlowEdge("1-2", "1", "2", animated=True),
            StreamlitFlowEdge("2-3", "2", "3", animated=True)
        ]
    if 'flow_key' not in st.session_state:
        st.session_state['flow_key'] = f'hackable_flow_{random.randint(0, 1000)}'
    if 'layout' not in st.session_state:
        st.session_state['layout'] = RadialLayout()
    if 'selected_node_id' not in st.session_state:
        st.session_state['selected_node_id'] = None

    with st.sidebar:
        st.subheader("Workflow Actions")
        
        st.subheader("Add New Node")
        node_type = st.selectbox("Select node type", ["Input", "LLM", "Output"])
        if st.button("Add Node"):
            new_node_id = str(len(st.session_state['nodes']) + 1)
            new_node = create_node(new_node_id, node_type, (random.randint(0, 600), random.randint(0, 400)))
            st.session_state['nodes'].append(new_node)
            st.rerun()
        
        st.subheader("Select Layout")
        layout_options = {
            "TreeLayout": TreeLayout(direction='down'),
            "RadialLayout": RadialLayout(),
            "ForceLayout": ForceLayout(),
            "StressLayout": StressLayout(),
            "RandomLayout": RandomLayout(),
        }
        selected_layout = st.selectbox("Select layout", list(layout_options.keys()), key="layout_select")
        st.session_state['layout'] = layout_options[selected_layout]

        st.subheader("Configure Node")
        selected_node_id = st.selectbox("Select Node to Configure", [node.id for node in st.session_state['nodes']])
        selected_node = next((node for node in st.session_state['nodes'] if node.id == selected_node_id), None)
        if selected_node:
            render_node_settings(selected_node)

    st.subheader("Workflow Canvas")

    # Generate a new key when loading to force a refresh
    if st.button("Load Workflow"):
        loaded_nodes, loaded_edges = load_workflow()
        if loaded_nodes:
            st.session_state['nodes'] = loaded_nodes
            st.session_state['edges'] = loaded_edges
            st.session_state['flow_key'] = f'hackable_flow_{random.randint(0, 1000)}'  # Generate new key
            st.success("Workflow loaded successfully!")
            st.rerun()
        else:
            st.warning("No saved workflow found.")

    flow_response = streamlit_flow(
        st.session_state['flow_key'],
        st.session_state['nodes'],
        st.session_state['edges'],
        layout=st.session_state['layout'],
        fit_view=True,
        height=600,
        enable_node_menu=True,
        show_minimap=True,
        enable_pane_menu=True,
        hide_watermark=True,
        allow_new_edges=True,
        min_zoom=0.1
    )

    if flow_response and "edges" in flow_response:
        st.session_state['edges'] = [StreamlitFlowEdge.from_dict(edge) for edge in flow_response["edges"]]

    st.subheader("Workflow Controls")
    col1, col2, col4 = st.columns(3)
    with col1:
        if st.button("Execute Workflow"):
            with st.spinner("Executing workflow..."):
                results = execute_workflow(st.session_state['nodes'], st.session_state['edges'])
            st.write("Workflow Execution Results:")
            for node_id, result in results.items():
                node = next(node for node in st.session_state['nodes'] if node.id == node_id)
                st.subheader(f"Node {node_id} ({node.data['type']}):")
                if node.data['type'] == 'Output':
                    if node.data['document_format'] == 'Markdown':
                        st.markdown(result)
                    elif node.data['document_format'] == 'HTML':
                        st.components.v1.html(result, height=300)
                    else:
                        st.text(result)
                else:
                    st.text(result)
    with col2:
        if st.button("Save Workflow"):
            save_workflow(st.session_state['nodes'], st.session_state['edges'])
            st.success("Workflow saved successfully!")
    with col4:
        if st.button("Clear Workflow"):
            st.session_state['nodes'] = []
            st.session_state['edges'] = []
            st.session_state['selected_node_id'] = None
            st.session_state['flow_key'] = f'hackable_flow_{random.randint(0, 1000)}'  # Generate new key
            st.success("Workflow cleared successfully!")
            st.rerun()

    # Display current workflow information
    st.subheader("Current Workflow Information")
    st.write(f"Number of Nodes: {len(st.session_state['nodes'])}")
    st.write(f"Number of Edges: {len(st.session_state['edges'])}")
    
    # Display node information
    st.subheader("Node Information")
    for node in st.session_state['nodes']:
        with st.expander(f"Node {node.id} ({node.data['type']})"):
            st.write(f"Content: {node.data['content']}")
            st.write(f"Type: {node.data['type']}")
            if 'pos' in node.data:
                st.write(f"Position: {node.data['pos']}")
            if node.data['type'] == 'LLM':
                st.write(f"Model: {node.data['model_name']}")
                st.write(f"Agent Type: {node.data['agent_type']}")
                st.write(f"Metacognitive Type: {node.data['metacognitive_type']}")
                st.write(f"Voice Type: {node.data['voice_type']}")
                st.write(f"Identity Type: {node.data['identity_type']}")
                st.write(f"Temperature: {node.data['temperature']}")
                st.write(f"Max Tokens: {node.data['max_tokens']}")
                st.write(f"Presence Penalty: {node.data['presence_penalty']}")
                st.write(f"Frequency Penalty: {node.data['frequency_penalty']}")
                st.text_area("Prompt", value=node.data['prompt'], disabled=True)
            elif node.data['type'] == 'Input':
                st.text_area("Input Text", value=node.data['input_text'], disabled=True)
            elif node.data['type'] == 'Output':
                st.write(f"Output Label: {node.data['output_label']}")
                st.write(f"Document Format: {node.data['document_format']}")

    # Display edge information
    st.subheader("Edge Information")
    for edge in st.session_state['edges']:
        st.write(f"Edge: {edge.source} -> {edge.target}")

if __name__ == "__main__":
    nodes_interface()