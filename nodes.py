# nodes.py

import json
import logging
import streamlit as st
import requests
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Union, Tuple
from ollama_utils import get_available_models, call_ollama_endpoint, load_api_keys
from openai_utils import OPENAI_MODELS, call_openai_api
from groq_utils import GROQ_MODELS, call_groq_api
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define available node types
AVAILABLE_NODE_TYPES = {
    "Input": True,
    "Processing": True,
    "LLM": True,
    "Output": True,
    "DataRetrieval": False,  # Set to False until fully implemented
    "Control": False,  # Set to False until fully implemented
    "Integration": False,  # Set to False until fully implemented
    "Utility": False  # Set to False until fully implemented
}

# Define color scheme and emojis
NODE_COLORS = {
    "Input": "#90EE90",
    "LLM": "#ADD8E6",
    "Output": "#FFB6C1",
    "Processing": "#FFD700",
    "DataRetrieval": "#FFA07A",
    "Control": "#98FB98",
    "Integration": "#DDA0DD",
    "Utility": "#87CEFA"
}

NODE_EMOJIS = {
    "Input": "⤵️",
    "LLM": "🧠",
    "Output": "⤴️",
    "Processing": "⚙️",
    "DataRetrieval": "🔍",
    "Control": "🔀",
    "Integration": "🔌",
    "Utility": "🛠️"
}

class Node:
    """Represents a node in the workflow."""
    def __init__(self, id: str, node_type: str, data: dict):
        """
        Initializes a Node object.

        Args:
            id (str): The unique identifier of the node.
            node_type (str): The type of the node (e.g., 'Input', 'LLM', 'Output').
            data (dict): A dictionary containing the data and configuration of the node.
        """
        self.id = id
        self.type = node_type
        self.data = data

    def to_dict(self):
        """Converts the Node object to a dictionary."""
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data
        }

    @staticmethod
    def from_dict(data: dict):
        """Creates a Node object from a dictionary."""
        node_type = data['type']
        default_data = create_node("temp", node_type).data
        merged_data = {**default_data, **data.get('data', {})}
        return Node(data['id'], data['type'], merged_data)

class Edge:
    """Represents an edge connecting two nodes in the workflow."""
    def __init__(self, id: str, source: str, target: str):
        """
        Initializes an Edge object.

        Args:
            id (str): The unique identifier of the edge.
            source (str): The ID of the source node.
            target (str): The ID of the target node.
        """
        self.id = id
        self.source = source
        self.target = target

    def to_dict(self):
        """Converts the Edge object to a dictionary."""
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target
        }

    @staticmethod
    def from_dict(data: dict):
        """Creates an Edge object from a dictionary."""
        try:
            return Edge(data['id'], data['source'], data['target'])
        except KeyError as e:
            raise KeyError(f"Missing key {e} in edge data: {data}")

def get_all_models() -> list:
    """Retrieves a list of all available models."""
    ollama_models = get_available_models()
    all_models = ollama_models + OPENAI_MODELS + GROQ_MODELS
    return all_models

def call_api_and_decode_response(api_function, *args, **kwargs) -> dict:
    """Calls the specified API function, decodes the JSON response, and validates its structure."""
    try:
        response = api_function(*args, **kwargs)
        
        if not response or not isinstance(response, str) or response.strip() == "":
            raise ValueError("Received empty or invalid response from the API")

        try:
            workflow_data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {str(e)} - Response content: {response}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        
        if 'nodes' not in workflow_data or 'edges' not in workflow_data:
            raise ValueError("Invalid JSON structure: Missing 'nodes' or 'edges'")

        for edge in workflow_data.get('edges', []):
            if 'source' not in edge or 'target' not in edge:
                raise ValueError(f"Missing key 'source' or 'target' in edge data: {edge}")
        
        return workflow_data
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def generate_workflow(user_request: str, model: str) -> Tuple[List[Node], List[Edge]]:
    """Generates a workflow based on the user's request using the specified language model."""
    available_types = [node_type for node_type, available in AVAILABLE_NODE_TYPES.items() if available]
    
    prompt = f"""
    Generate a workflow to accomplish the following task:
    {user_request}

    You can use the following node types:
    {', '.join(available_types)}

    Provide a JSON output with the following structure:
    {{
        "nodes": [
            {{
                "id": "1",
                "type": "Input",
                "data": {{
                    "content": "Input Node",
                    "input_type": "Text",
                    "input_text": ""
                }}
            }},
            ...
        ],
        "edges": [
            {{
                "id": "1-2",
                "source": "1",
                "target": "2"
            }},
            ...
        ]
    }}

    Ensure that the workflow starts with an Input node and ends with an Output node.
    Include appropriate nodes as needed, using only the node types listed above.
    Provide detailed configurations for each node, including prompts for LLM nodes and specific settings for other node types.
    """

    api_keys = load_api_keys()

    try:
        if model in OPENAI_MODELS:
            response = call_openai_api(
                model,
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                openai_api_key=api_keys.get("openai_api_key")
            )
            # OpenAI models return the response directly, so we need to parse it
            workflow_data = parse_openai_response(response)
        elif model in GROQ_MODELS:
            response = call_groq_api(
                model,
                prompt,
                temperature=0.7,
                max_tokens=2000,
                groq_api_key=api_keys.get("groq_api_key")
            )
            # Groq models might need similar parsing as OpenAI
            workflow_data = parse_openai_response(response)
        else:
            response, _, _, _ = call_ollama_endpoint(
                model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            # Ollama models might return a different format, so we'll parse it differently
            workflow_data = parse_ollama_response(response)
        
        nodes = [Node.from_dict(node_data) for node_data in workflow_data["nodes"]]
        edges = [Edge.from_dict(edge_data) for edge_data in workflow_data["edges"]]
        return nodes, edges
    except Exception as e:
        st.error(f"Failed to generate a valid workflow: {str(e)}")
        return [], []

def parse_openai_response(response: str) -> Dict:
    """Parses the response from OpenAI models."""
    try:
        # Find the JSON object in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end]
            workflow_data = json.loads(json_str)
            if 'nodes' not in workflow_data or 'edges' not in workflow_data:
                raise ValueError("Invalid JSON structure: Missing 'nodes' or 'edges'")
            return workflow_data
        else:
            raise ValueError("No valid JSON object found in the response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in OpenAI response: {str(e)}")

def parse_ollama_response(response: str) -> Dict:
    """Parses the response from Ollama models."""
    try:
        # Ollama might return the entire conversation, so we need to extract the last message
        messages = response.split('\n\n')
        last_message = messages[-1]
        
        # Find the JSON object in the last message
        json_start = last_message.find('{')
        json_end = last_message.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = last_message[json_start:json_end]
            workflow_data = json.loads(json_str)
            if 'nodes' not in workflow_data or 'edges' not in workflow_data:
                raise ValueError("Invalid JSON structure: Missing 'nodes' or 'edges'")
            
            # Ensure each edge has 'source' and 'target'
            for edge in workflow_data['edges']:
                if 'source' not in edge or 'target' not in edge:
                    edge['source'] = edge.get('source', edge['id'].split('-')[0])
                    edge['target'] = edge.get('target', edge['id'].split('-')[1])
            
            return workflow_data
        else:
            raise ValueError("No valid JSON object found in the Ollama response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in Ollama response: {str(e)}")

def execute_workflow(nodes: List[Node], edges: List[Edge]) -> Dict[str, str]:
    """Executes the workflow defined by the nodes and edges."""
    results = {}
    node_map = {node.id: node for node in nodes}
    api_keys = load_api_keys()

    def process_node(node_id: str) -> str:
        """Recursively processes a node and its inputs."""
        if node_id in results:
            return results[node_id]

        if node_id not in node_map:
            logger.error(f"Node {node_id} not found in workflow")
            return f"Error: Node {node_id} not found"

        node = node_map[node_id]
        incoming_edges = [edge for edge in edges if edge.target == node_id]

        try:
            if node.type == 'Input':
                results[node_id] = handle_input_node(node)
            elif node.type == 'Processing':
                results[node_id] = handle_processing_node(node, incoming_edges, process_node)
            elif node.type == 'LLM':
                results[node_id] = handle_llm_node(node, incoming_edges, process_node, api_keys)
            elif node.type == 'Output':
                results[node_id] = handle_output_node(node, incoming_edges, process_node)
            else:
                logger.error(f"Unsupported node type: {node.type}")
                results[node_id] = f"Error: Unsupported node type {node.type}"
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            results[node_id] = f"Error processing node {node_id}: {str(e)}"

        return results[node_id]

    for node in nodes:
        process_node(node.id)

    return results

def handle_input_node(node: Node) -> str:
    """Handles the execution of an Input node."""
    if node.data['input_type'] == 'Text':
        return node.data['input_text']
    elif node.data['input_type'] == 'File':
        if node.data['file_upload'] is not None:
            return node.data['file_upload'].getvalue().decode('utf-8')
        else:
            return "Error: No file uploaded"
    elif node.data['input_type'] == 'API':
        try:
            response = requests.get(node.data['api_endpoint'])
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error fetching API: {str(e)}"
    else:
        return f"Unknown input type: {node.data['input_type']}"

def handle_processing_node(node: Node, incoming_edges: List[Edge], process_node) -> str:
    """Handles the execution of a Processing node."""
    if not incoming_edges:
        return "Error: Processing node requires input"
    input_data = process_node(incoming_edges[0].source)

    if node.data['processing_type'] == 'Preprocessing':
        for step in node.data['preprocessing_steps']:
            if step == 'Tokenization':
                input_data = input_data.split()
            elif step == 'Lowercasing':
                input_data = input_data.lower()
            elif step == 'Remove Punctuation':
                input_data = ''.join(char for char in input_data if char.isalnum() or char.isspace())
            elif step == 'Remove Stopwords':
                stopwords = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of'])
                input_data = ' '.join([word for word in input_data.split() if word.lower() not in stopwords])
        return ' '.join(input_data) if isinstance(input_data, list) else input_data

    elif node.data['processing_type'] == 'Logic':
        logic_type = node.data['logic_type']
        if logic_type == 'Monte Carlo Tree Search':
            return execute_mcts_subworkflow(node)
        elif logic_type in ['Chain of Thought', 'Tree of Thought', 'Visualization of Thought']:
            return apply_metacognitive_prompt(node, input_data)
        else:
            return f"Unknown logic type: {logic_type}"

    elif node.data['processing_type'] == 'Vectorization':
        return f"Vectorized: {input_data[:50]}..."
    else:
        return f"Unknown processing type: {node.data['processing_type']}"

def handle_llm_node(node: Node, incoming_edges: List[Edge], process_node, api_keys: Dict[str, str]) -> str:
    """Handles the execution of an LLM node."""
    if not incoming_edges:
        return "Error: LLM node requires input"
    input_text = process_node(incoming_edges[0].source)
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

    # Append to conversation history
    node.data['conversation_history'].append({"role": "user", "content": input_text})
    node.data['conversation_history'].append({"role": "assistant", "content": response})

    return response

def handle_output_node(node: Node, incoming_edges: List[Edge], process_node) -> str:
    """Handles the execution of an Output node."""
    if not incoming_edges:
        return "Error: Output node requires input"
    input_data = process_node(incoming_edges[0].source)
    if node.data['output_type'] == 'Text':
        return f"{node.data['output_label']}: {input_data}"
    elif node.data['output_type'] == 'File':
        return input_data  # Return the raw data for file download
    elif node.data['output_type'] == 'Visualization':
        plt.figure(figsize=(10, 5))
        plt.plot([1, 2, 3, 4, 5], [1, 4, 2, 3, 5])
        plt.title(input_data[:30])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    else:
        return f"Unknown output type: {node.data['output_type']}"

def construct_prompt(node: Node, base_prompt: str, input_text: str) -> str:
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
    
    if node.data['conversation_history']:
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in node.data['conversation_history'][-5:]])
        prompt_parts.append(f"Recent Conversation:\n{history}")
    
    return "\n\n".join(prompt_parts)

def validate_workflow(nodes: List[Node], edges: List[Edge]) -> Dict[str, Union[bool, str]]:
    """Validates the workflow for structural correctness."""
    if not any(node.type == 'Input' for node in nodes):
        return {'valid': False, 'error': 'Workflow must have at least one Input node.'}

    if not any(node.type == 'Output' for node in nodes):
        return {'valid': False, 'error': 'Workflow must have at least one Output node.'}

    node_ids = set(node.id for node in nodes)
    connected_nodes = set()
    for edge in edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            return {'valid': False, 'error': f'Edge {edge.id} connects to non-existent nodes: {edge.source} -> {edge.target}'}
        connected_nodes.add(edge.source)
        connected_nodes.add(edge.target)
    if node_ids != connected_nodes:
        return {'valid': False, 'error': 'All nodes must be connected in the workflow.'}

    if has_cycle(nodes, edges):
        return {'valid': False, 'error': 'Workflow contains a cycle. It must be acyclic.'}

    if not path_exists_input_to_output(nodes, edges):
        return {'valid': False, 'error': 'There must be a path from an Input node to an Output node.'}

    return {'valid': True, 'error': None}

def has_cycle(nodes: List[Node], edges: List[Edge]) -> bool:
    """Checks if the workflow graph contains a cycle."""
    graph = {node.id: set() for node in nodes}
    for edge in edges:
        graph[edge.source].add(edge.target)

    visited = set()
    rec_stack = set()

    def is_cyclic(node_id: str) -> bool:
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
    """Checks if there is a path from an Input node to an Output node."""
    graph = {node.id: set() for node in nodes}
    for edge in edges:
        graph[edge.source].add(edge.target)

    input_nodes = [node.id for node in nodes if node.type == 'Input']
    output_nodes = [node.id for node in nodes if node.type == 'Output']

    def dfs(node_id: str, visited: set) -> bool:
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

def execute_mcts_subworkflow(node: Node) -> str:
    """Executes a sub-workflow as part of MCTS logic."""
    # Placeholder for MCTS logic; this would involve creating and managing a sub-workflow.
    # For demonstration purposes, we'll mock the MCTS behavior.
    return "MCTS decision result"

def apply_metacognitive_prompt(node: Node, input_data: str) -> str:
    """Applies a metacognitive prompt for Chain of Thought, Tree of Thought, or Visualization of Thought."""
    metacognitive_type = node.data.get('metacognitive_type', 'None')
    prompt = get_metacognitive_prompt()[metacognitive_type]
    # Normally, we'd interact with the LLM API here using the prompt and input_data.
    # For now, we'll mock the LLM response.
    return f"{metacognitive_type} result based on input: {input_data}"

def render_node_settings(node: Node) -> None:
    """Renders the settings panel for the selected node in the Streamlit sidebar."""
    st.sidebar.subheader(f"Configure {NODE_EMOJIS[node.type]} {node.type} Node {node.id}")
    node.data['content'] = st.sidebar.text_input("Node Label", value=node.data['content'], key=f"label_{node.id}")

    if node.type == 'Input':
        node.data['input_type'] = st.sidebar.selectbox("Input Type", ["Text", "File", "API"], key=f"input_type_{node.id}")
        if node.data['input_type'] == 'Text':
            node.data['input_text'] = st.sidebar.text_area("Input Text", value=node.data['input_text'], key=f"input_text_{node.id}")
        elif node.data['input_type'] == 'File':
            node.data['file_upload'] = st.sidebar.file_uploader("Upload File", key=f"file_upload_{node.id}")
        elif node.data['input_type'] == 'API':
            node.data['api_endpoint'] = st.sidebar.text_input("API Endpoint", value=node.data['api_endpoint'], key=f"api_endpoint_{node.id}")

    elif node.type == 'Processing':
        node.data['processing_type'] = st.sidebar.selectbox("Processing Type", ["Preprocessing", "Logic", "Vectorization"], key=f"processing_type_{node.id}")
        if node.data['processing_type'] == 'Preprocessing':
            node.data['preprocessing_steps'] = st.sidebar.multiselect("Preprocessing Steps", ["Tokenization", "Lowercasing", "Remove Punctuation", "Remove Stopwords"], key=f"preprocessing_steps_{node.id}")
        elif node.data['processing_type'] == 'Vectorization':
            node.data['vectorization_model'] = st.sidebar.selectbox("Vectorization Model", ["Word2Vec", "GloVe", "FastText"], key=f"vectorization_model_{node.id}")
        elif node.data['processing_type'] == 'Logic':
            metacognitive_types = list(get_metacognitive_prompt().keys())
            logic_types = ["Monte Carlo Tree Search"] + metacognitive_types
            node.data['logic_type'] = st.sidebar.selectbox("Logic Type", logic_types, key=f"logic_type_{node.id}")

    elif node.type == 'LLM':
        all_models = get_all_models()
        node.data['model_name'] = st.sidebar.selectbox(
            "Select Model", 
            all_models, 
            index=all_models.index(node.data['model_name']) if node.data['model_name'] in all_models else 0, 
            key=f"model_{node.id}"
        )
        node.data['agent_type'] = st.sidebar.selectbox("Agent Type", ["None"] + list(get_agent_prompt().keys()), index=(["None"] + list(get_agent_prompt().keys())).index(node.data['agent_type']), key=f"agent_type_{node.id}")
        node.data['voice_type'] = st.sidebar.selectbox("Voice Type", ["None"] + list(get_voice_prompt().keys()), index=(["None"] + list(get_voice_prompt().keys())).index(node.data['voice_type']), key=f"voice_type_{node.id}")
        node.data['identity_type'] = st.sidebar.selectbox("Identity Type", ["None"] + list(get_identity_prompt().keys()), index=(["None"] + list(get_identity_prompt().keys())).index(node.data['identity_type']), key=f"identity_type_{node.id}")
        node.data['temperature'] = st.sidebar.slider("Temperature", 0.0, 1.0, node.data['temperature'], key=f"temperature_{node.id}")
        node.data['max_tokens'] = st.sidebar.slider("Max Tokens", 1000, 128000, node.data['max_tokens'], step=1000, key=f"max_tokens_{node.id}")
        node.data['presence_penalty'] = st.sidebar.slider("Presence Penalty", -2.0, 2.0, node.data['presence_penalty'], step=0.1, key=f"presence_penalty_{node.id}")
        node.data['frequency_penalty'] = st.sidebar.slider("Frequency Penalty", -2.0, 2.0, node.data['frequency_penalty'], step=0.1, key=f"frequency_penalty_{node.id}")
        node.data['prompt'] = st.sidebar.text_area("Prompt", value=node.data['prompt'], key=f"prompt_input_{node.id}")
        node.data['fine_tuning'] = st.sidebar.checkbox("Enable Fine-tuning", value=node.data['fine_tuning'], key=f"fine_tuning_{node.id}")

    elif node.type == 'Output':
        node.data['output_type'] = st.sidebar.selectbox("Output Type", ["Text", "File", "Visualization"], key=f"output_type_{node.id}")
        node.data['output_label'] = st.sidebar.text_input("Output Label", value=node.data['output_label'], key=f"output_label_{node.id}")
        if node.data['output_type'] == 'Text':
            node.data['document_format'] = st.sidebar.selectbox("Document Format", ["Text", "Markdown", "HTML"], index=["Text", "Markdown", "HTML"].index(node.data['document_format']), key=f"document_format_{node.id}")
        elif node.data['output_type'] == 'File':
            node.data['file_format'] = st.sidebar.selectbox("File Format", ["txt", "csv", "json"], key=f"file_format_{node.id}")
        elif node.data['output_type'] == 'Visualization':
            node.data['visualization_type'] = st.sidebar.selectbox("Visualization Type", ["Bar Chart", "Line Chart", "Scatter Plot"], key=f"visualization_type_{node.id}")

    if st.sidebar.button("Update Node", key=f"update_node_{node.id}"):
        st.success(f"Node {node.id} updated successfully!")
        st.rerun()



def render_workflow_canvas(nodes: List[Node], edges: List[Edge]) -> None:
    """Renders the workflow canvas, displaying nodes and connections."""
    cols = st.columns(3)
    for i, node in enumerate(nodes):
        with cols[i % 3]:
            with st.container():
                if st.button(f"{NODE_EMOJIS[node.type]} Node {node.id}: {node.type}", 
                             key=f"node_button_{node.id}",
                             use_container_width=True):
                    st.session_state.selected_node_id = node.id
                    st.rerun()
                
                st.markdown(f"<div style='background-color: {NODE_COLORS[node.type]}; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
                st.write(f"**Content:** {node.data.get('content', 'No content specified')}")
                if node.type == 'Input':
                    st.write(f"**Input Type:** {node.data.get('input_type', 'Not specified')}")
                elif node.type == 'Processing':
                    st.write(f"**Processing Type:** {node.data.get('processing_type', 'Not specified')}")
                elif node.type == 'LLM':
                    st.write(f"**Model:** {node.data.get('model_name', 'Not specified')}")
                elif node.type == 'Output':
                    st.write(f"**Output Type:** {node.data.get('output_type', 'Not specified')}")
                st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🔀 Connections")
    for edge in edges:
        st.markdown(f"<div style='text-align: center; font-weight: bold;'>Node {edge.source} ➡️ Node {edge.target}</div>", unsafe_allow_html=True)
        
def nodes_interface() -> None:
    """Provides the Streamlit interface for the LLM workflow builder."""
    st.title("✳️ Compound Elemental Framework (CEF)")

    st.markdown("""
    <style>
    div.stButton > button:first-child.stBtn.secondary {
        background-color: blue;
        color: white;
    }
    div.stButton > button:hover:first-child.stBtn.secondary {
        background-color: red;
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

    mode = st.radio("Select mode:", ["Manual", "AI-Assisted"])

    if mode == "AI-Assisted":
        user_request = st.text_input("Enter your request for the AI to generate a workflow:")
        workflow_model = st.selectbox("Select model for workflow generation:", get_all_models())
        if st.button("Generate Workflow"):
            with st.spinner("Generating workflow..."):
                nodes, edges = generate_workflow(user_request, workflow_model)
                if nodes and edges:
                    st.session_state['nodes'] = nodes
                    st.session_state['edges'] = edges
                    st.success("Workflow generated successfully!")
                    st.rerun()

    with st.sidebar:
        st.subheader("🛠️ Workflow Actions")
        
        st.subheader("➕ Add New Node")
        available_node_types = [node_type for node_type, available in AVAILABLE_NODE_TYPES.items() if available]
        node_type = st.selectbox("Select node type", available_node_types)
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
    if st.button("▶️ Execute Workflow", type="primary"):
        validation_result = validate_workflow(st.session_state['nodes'], st.session_state['edges'])
        if validation_result['valid']:
            with st.spinner("Executing workflow..."):
                results = execute_workflow(st.session_state['nodes'], st.session_state['edges'])
            st.write("Workflow Execution Results:")
            for node_id, result in results.items():
                node = next(node for node in st.session_state['nodes'] if node.id == node_id)
                st.subheader(f"{NODE_EMOJIS[node.type]} Node {node_id} ({node.type}):")
                if node.type == 'Output':
                    if node.data['output_type'] == 'Text':
                        if node.data['document_format'] == 'Markdown':
                            st.markdown(result)
                        elif node.data['document_format'] == 'HTML':
                            st.components.v1.html(result, height=300)
                        else:
                            st.text(result)
                    elif node.data['output_type'] == 'File':
                        file_contents = result.encode('utf-8')
                        st.download_button(
                            label="Download Output File",
                            data=file_contents,
                            file_name=f"output.{node.data['file_format']}",
                            mime=f"text/{node.data['file_format']}"
                        )
                    elif node.data['output_type'] == 'Visualization':
                        st.image(result)
                else:
                    st.text(result)
        else:
            st.error(f"Workflow validation failed: {validation_result['error']}")

    st.html("<hr />")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✅ Validate Workflow", type="primary"):
            validation_result = validate_workflow(st.session_state['nodes'], st.session_state['edges'])
            if validation_result['valid']:
                st.success("Workflow is valid! ✅")
            else:
                st.error(f"Workflow validation failed: {validation_result['error']}")
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
    with col4:
        if st.button("📤 Export as Script", type="secondary"):
            script = export_workflow_as_script(st.session_state['nodes'], st.session_state['edges'])
            st.download_button(
                label="Download Python Script",
                data=script,
                file_name="workflow_script.py",
                mime="text/plain"
            )


    st.subheader("📊 Workflow Statistics")
    st.write(f"Total Nodes: {len(st.session_state['nodes'])}")
    st.write(f"Total Edges: {len(st.session_state['edges'])}")

    node_types = {}
    for node in st.session_state['nodes']:
        node_types[node.type] = node_types.get(node.type, 0) + 1

    st.write("Node Types:")
    for node_type, count in node_types.items():
        st.write(f"  - {NODE_EMOJIS[node_type]} {node_type}: {count}")

    with st.expander("❓ Help"):
        st.markdown("""
        ### How to use the LLM Workflow Builder:
        1. **Add Nodes**: Use the sidebar to add various types of nodes (Input, Processing, LLM, Output).
        2. **Connect Nodes**: Add edges to connect nodes in the desired order.
        3. **Configure Nodes**: Click on a node in the canvas to configure its settings in the sidebar.
        4. **Execute Workflow**: Click the 'Execute Workflow' button to run your workflow.
        5. **Save/Load**: Save your workflow for later use or load a previously saved workflow.
        6. **Validate**: Use the 'Validate Workflow' button to check if your workflow is properly constructed.
        7. **Export**: Export your workflow as a Python script for external execution.

        ### AI-Assisted Mode:
        1. Select "AI-Assisted" mode.
        2. Enter your workflow request in natural language.
        3. Choose a model for workflow generation.
        4. Click "Generate Workflow" to create a workflow based on your request.
        5. Review and modify the generated workflow as needed.

        For more detailed instructions, please refer to the documentation.
        """)

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

def create_node(node_id: str, node_type: str) -> Node:
    """Creates a new node with default settings based on its type."""
    if not AVAILABLE_NODE_TYPES.get(node_type, False):
        raise ValueError(f"Node type '{node_type}' is not available or implemented.")
    
    base_data = {
        'content': f'{node_type} Node',
    }
    if node_type == 'Input':
        base_data.update({
            'input_type': 'Text',
            'input_text': '',
            'file_upload': None,
            'api_endpoint': '',
        })
    elif node_type == 'Processing':
        base_data.update({
            'processing_type': 'Preprocessing',
            'preprocessing_steps': [],
            'vectorization_model': 'default',
        })
    elif node_type == 'LLM':
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
            'fine_tuning': False,
            'conversation_history': [],
        })
    elif node_type == 'Output':
        base_data.update({
            'output_type': 'Text',
            'output_label': 'Output:',
            'document_format': 'Text',
            'file_format': 'txt',
            'visualization_type': 'None',
        })
    return Node(node_id, node_type, base_data)

def save_workflow(nodes: List[Node], edges: List[Edge]) -> None:
    """Saves the current workflow to a JSON file."""
    workflow_data = {
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges]
    }
    with open("workflow.json", "w") as f:
        json.dump(workflow_data, f, indent=2)

def load_workflow() -> Tuple[List[Node], List[Edge]]:
    """Loads a workflow from a JSON file."""
    try:
        with open("workflow.json", "r") as f:
            workflow_data = json.load(f)
        nodes = [Node.from_dict(node_data) for node_data in workflow_data["nodes"]]
        edges = [Edge.from_dict(edge_data) for edge_data in workflow_data["edges"]]
        return nodes, edges
    except FileNotFoundError:
        st.warning("No saved workflow found.")
        return [], []
    except json.JSONDecodeError:
        st.error("Error decoding the saved workflow. The file may be corrupted.")
        return [], []

def export_workflow_as_script(nodes: List[Node], edges: List[Edge]) -> str:
    """Exports the workflow as a Python script."""
    script = """
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_workflow():
    results = {}
    """

    for node in nodes:
        script += f"\n    # Node {node.id}: {node.type}\n"
        if node.type == 'Input':
            script += f"    results['{node.id}'] = {json.dumps(node.data['input_text'])}\n"
        elif node.type == 'Processing':
            script += f"    input_data = results['{edges[0].source}']\n"
            script += f"    # Add processing logic here\n"
            script += f"    results['{node.id}'] = input_data  # Placeholder\n"
        elif node.type == 'LLM':
            script += f"    input_data = results['{edges[0].source}']\n"
            script += f"    # Add LLM API call logic here\n"
            script += f"    results['{node.id}'] = 'LLM response'  # Placeholder\n"
        elif node.type == 'Output':
            script += f"    input_data = results['{edges[0].source}']\n"
            script += f"    print(f'Output: {{input_data}}')\n"

    script += "\n    return results\n\n"
    script += "if __name__ == '__main__':\n"
    script += "    execute_workflow()\n"

    return script

def main() -> None:
    """The main function to run the Streamlit app."""
    st.set_page_config(page_title="Compound Elemental Framework (CEF)", page_icon="✳️", layout="wide")
    nodes_interface()

if __name__ == "__main__":
    main()