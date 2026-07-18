# nodes.py

import ast
import json
import logging
import operator
import os
import re
import time
import streamlit as st
import requests
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Union, Tuple
from ollama_workbench.providers.ollama_utils import get_available_models, get_all_models, call_ollama_endpoint, load_api_keys
from ollama_workbench.providers.openai_utils import OPENAI_MODELS, call_openai_api, get_openai_models
from ollama_workbench.providers.groq_utils import GROQ_MODELS, call_groq_api, get_groq_models
from ollama_workbench.ui.prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt

logger = logging.getLogger(__name__)


def _safe_eval_condition(condition_str, input_data):
    """Safely evaluate a simple condition like '{input} > 5' or '{input} == "hello"'.

    Supports basic comparison operators only: ==, !=, >=, <=, >, <.
    No arbitrary code execution.
    """
    condition_str = condition_str.replace('{input}', repr(input_data))
    ops = {
        '==': operator.eq, '!=': operator.ne,
        '>=': operator.ge, '<=': operator.le,
        '>': operator.gt, '<': operator.lt,
    }
    for op_str, op_func in sorted(ops.items(), key=lambda x: -len(x[0])):
        if op_str in condition_str:
            left, right = condition_str.split(op_str, 1)
            try:
                left_val = ast.literal_eval(left.strip())
                right_val = ast.literal_eval(right.strip())
                return op_func(left_val, right_val)
            except (ValueError, SyntaxError):
                return False
    # No comparison operator found; try literal eval for truthy check
    try:
        return bool(ast.literal_eval(condition_str.strip()))
    except (ValueError, SyntaxError):
        return False


def _safe_eval_math(expr_str):
    """Safely evaluate a math expression containing only numeric literals and +, -, *, /.

    Parses the expression into an AST and evaluates it recursively.
    No eval(), no function calls, no attribute access, no name lookups.
    """
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty expression")
    tree = ast.parse(expr_str, mode='eval')

    _OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
    }

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Only numeric values allowed, got {type(node.value).__name__}")
            return node.value
        if isinstance(node, ast.BinOp):
            op_func = _OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_func(_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -_eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +_eval_node(node.operand)
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return _eval_node(tree)


# Define available node types
AVAILABLE_NODE_TYPES = {
    "Input": True,
    "Processing": True,
    "LLM": True,
    "Output": True,
    "DataRetrieval": True,  # Now implemented
    "Control": True,  # Now implemented
    "Integration": True,  # Now implemented
    "Utility": True  # Now implemented
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
        if model in get_openai_models():
            response = call_openai_api(
                model,
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                openai_api_key=api_keys.get("openai_api_key")
            )
            # OpenAI models return the response directly, so we need to parse it
            workflow_data = parse_openai_response(response)
        elif model in get_groq_models():
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
            response, _, _, _, _ = call_ollama_endpoint(
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
            results[node_id] = f"Error: Node {node_id} not found"
            return results[node_id]

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
            elif node.type == 'DataRetrieval':
                results[node_id] = handle_data_retrieval_node(node, incoming_edges, process_node)
            elif node.type == 'Control':
                results[node_id] = handle_control_node(node, incoming_edges, process_node)
            elif node.type == 'Integration':
                results[node_id] = handle_integration_node(node, incoming_edges, process_node)
            elif node.type == 'Utility':
                results[node_id] = handle_utility_node(node, incoming_edges, process_node)
            else:
                logger.error(f"Unsupported node type: {node.type}")
                results[node_id] = f"Error: Unsupported node type {node.type}"
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            results[node_id] = f"Error processing node {node_id}: {str(e)}"

        return results[node_id]

    # Find all output nodes and process them (which will trigger processing of all their inputs)
    output_nodes = [node.id for node in nodes if node.type == 'Output']
    for node_id in output_nodes:
        process_node(node_id)
    
    # Process any nodes that haven't been processed yet
    for node in nodes:
        if node.id not in results:
            process_node(node.id)

    # Surface edges that reference nodes missing from the workflow
    for edge in edges:
        for ref in (edge.source, edge.target):
            if ref not in results:
                process_node(ref)

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
    
    if node.data['model_name'] in get_openai_models():
        response = call_openai_api(
            node.data['model_name'], 
            [{"role": "user", "content": complete_prompt}],
            temperature=node.data['temperature'],
            max_tokens=node.data['max_tokens'],
            openai_api_key=api_keys.get("openai_api_key")
        )
    elif node.data['model_name'] in get_groq_models():
        response = call_groq_api(
            node.data['model_name'],
            complete_prompt,
            temperature=node.data['temperature'],
            max_tokens=node.data['max_tokens'],
            groq_api_key=api_keys.get("groq_api_key")
        )
    else:
        response, _, _, _, _ = call_ollama_endpoint(
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

def handle_data_retrieval_node(node: Node, incoming_edges: List[Edge], process_node) -> str:
    """Handles the execution of a DataRetrieval node."""
    # Processing input if available
    input_text = ""
    if incoming_edges:
        input_text = process_node(incoming_edges[0].source)
    
    if node.data['retrieval_type'] == 'Database':
        # Simulate database retrieval
        database_type = node.data.get('database_type', 'SQL')
        query = node.data.get('query', '')
        if input_text:
            query = query.replace("{input}", input_text)
        
        if database_type == 'SQL':
            return f"SQL query result for: {query}"
        elif database_type == 'NoSQL':
            return f"NoSQL query result for: {query}"
        elif database_type == 'Vector':
            return f"Vector database search result for: {query}"
        else:
            return f"Unknown database type: {database_type}"
    
    elif node.data['retrieval_type'] == 'API':
        # Make an actual API call if configured
        endpoint = node.data.get('api_endpoint', '')
        method = node.data.get('api_method', 'GET')
        params = node.data.get('api_params', {})
        
        # Replace any input placeholders in the params
        for key, value in params.items():
            if isinstance(value, str) and '{input}' in value:
                params[key] = value.replace('{input}', input_text)
        
        try:
            if method == 'GET':
                response = requests.get(endpoint, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(endpoint, json=params, timeout=10)
            else:
                return f"Unsupported API method: {method}"
            
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"API request error: {str(e)}"
    
    elif node.data['retrieval_type'] == 'File':
        # Read from a file
        file_path = node.data.get('file_path', '')
        if not file_path:
            return "No file path specified"
        # Restrict file access to the application directory
        app_dir = os.path.realpath(os.getcwd())
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(app_dir):
            return f"Access denied: file path must be within the application directory ({app_dir})"
        if not os.path.isfile(real_path):
            return f"File not found: {file_path}"

        try:
            with open(real_path, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"File read error: {str(e)}"
    
    elif node.data['retrieval_type'] == 'Web':
        # Extract data from a web page
        url = node.data.get('web_url', '')
        if input_text and not url:
            url = input_text
        
        if not url:
            return "Error: No URL specified"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Web request error: {str(e)}"
    
    else:
        return f"Unknown retrieval type: {node.data['retrieval_type']}"

def handle_control_node(node: Node, incoming_edges: List[Edge], process_node) -> str:
    """Handles the execution of a Control node."""
    if not incoming_edges:
        return "Error: Control node requires input"
    
    input_data = process_node(incoming_edges[0].source)
    
    if node.data['control_type'] == 'Conditional':
        # Implement conditional logic
        condition = node.data.get('condition', '')
        true_branch = node.data.get('true_branch', '')
        false_branch = node.data.get('false_branch', '')
        
        # Evaluate the condition based on input using safe comparison
        try:
            condition_result = _safe_eval_condition(condition, input_data)

            if condition_result:
                return f"TRUE: {true_branch}"
            else:
                return f"FALSE: {false_branch}"
        except Exception as e:
            return f"Condition evaluation error: {str(e)}"
    
    elif node.data['control_type'] == 'Loop':
        # Simulate a loop operation
        iterations = node.data.get('iterations', 1)
        loop_body = node.data.get('loop_body', '')
        
        results = []
        for i in range(iterations):
            # Replace placeholders in the loop body
            current_iteration = loop_body.replace('{i}', str(i)).replace('{input}', input_data)
            results.append(f"Iteration {i+1}: {current_iteration}")
        
        return "\n".join(results)
    
    elif node.data['control_type'] == 'Switch':
        # Implement a switch statement
        cases = node.data.get('cases', {})
        default_case = node.data.get('default_case', '')
        
        if input_data in cases:
            return f"CASE '{input_data}': {cases[input_data]}"
        else:
            return f"DEFAULT: {default_case}"
    
    else:
        return f"Unknown control type: {node.data['control_type']}"

def handle_integration_node(node: Node, incoming_edges: List[Edge], process_node) -> str:
    """Handles the execution of an Integration node."""
    if not incoming_edges:
        return "Error: Integration node requires input"
    
    input_data = process_node(incoming_edges[0].source)
    
    if node.data['integration_type'] == 'Email':
        # Simulate sending an email
        recipient = node.data.get('email_recipient', '')
        subject = node.data.get('email_subject', '')
        body = node.data.get('email_body', '').replace('{input}', input_data)
        
        # In a real implementation, you'd send an actual email
        # Here we just return a confirmation message
        message = MIMEMultipart()
        message['To'] = recipient
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        
        return f"Email would be sent to {recipient} with subject '{subject}' and body containing input data"
    
    elif node.data['integration_type'] == 'Webhook':
        # Make a webhook call
        webhook_url = node.data.get('webhook_url', '')
        webhook_method = node.data.get('webhook_method', 'POST')
        webhook_data = node.data.get('webhook_data', {})
        
        # Replace any input placeholders
        for key, value in webhook_data.items():
            if isinstance(value, str) and '{input}' in value:
                webhook_data[key] = value.replace('{input}', input_data)
        
        try:
            if webhook_method == 'POST':
                response = requests.post(webhook_url, json=webhook_data, timeout=10)
            elif webhook_method == 'GET':
                response = requests.get(webhook_url, params=webhook_data, timeout=10)
            else:
                return f"Unsupported webhook method: {webhook_method}"
            
            response.raise_for_status()
            return f"Webhook call successful: {response.text[:100]}"
        except requests.RequestException as e:
            return f"Webhook call error: {str(e)}"
    
    elif node.data['integration_type'] == 'Database':
        # Simulate database integration
        database_action = node.data.get('database_action', 'query')
        database_data = node.data.get('database_data', '').replace('{input}', input_data)
        
        return f"Database {database_action} with data: {database_data}"
    
    else:
        return f"Unknown integration type: {node.data['integration_type']}"

def handle_utility_node(node: Node, incoming_edges: List[Edge], process_node) -> str:
    """Handles the execution of a Utility node."""
    if not incoming_edges:
        return "Error: Utility node requires input"
    
    input_data = process_node(incoming_edges[0].source)
    
    if node.data['utility_type'] == 'Format':
        # Format the input data
        format_type = node.data.get('format_type', 'JSON')
        
        if format_type == 'JSON':
            try:
                # If input is a JSON string, parse it and re-format
                # If input is not JSON, try to convert simple data to JSON
                try:
                    data = json.loads(input_data)
                    return json.dumps(data, indent=2)
                except json.JSONDecodeError:
                    # Try to convert simple text to JSON
                    return json.dumps({"text": input_data}, indent=2)
            except Exception as e:
                return f"JSON formatting error: {str(e)}"
        
        elif format_type == 'HTML':
            # Simple HTML wrapping
            return f"<html><body><p>{input_data}</p></body></html>"
        
        elif format_type == 'Markdown':
            # Simple markdown formatting - wrap in code block
            return f"```\n{input_data}\n```"
        
        else:
            return f"Unknown format type: {format_type}"
    
    elif node.data['utility_type'] == 'Transform':
        # Apply transformations to input data
        transform_type = node.data.get('transform_type', 'Uppercase')
        
        if transform_type == 'Uppercase':
            return input_data.upper()
        elif transform_type == 'Lowercase':
            return input_data.lower()
        elif transform_type == 'Capitalize':
            return input_data.capitalize()
        elif transform_type == 'Count':
            return f"Character count: {len(input_data)}"
        else:
            return f"Unknown transform type: {transform_type}"
    
    elif node.data['utility_type'] == 'Calculate':
        # Perform calculations
        calculation_type = node.data.get('calculation_type', 'Basic Math')
        formula = node.data.get('formula', '')
        
        if calculation_type == 'Basic Math':
            try:
                result = _safe_eval_math(formula.replace('{input}', input_data))
                return f"Calculation result: {result}"
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        elif calculation_type == 'Statistics':
            try:
                # Parse the input data as a list of numbers
                try:
                    numbers = [float(n) for n in input_data.split()]
                except ValueError:
                    return "Invalid input for statistics: expecting space-separated numbers"
                
                import numpy as np
                return {
                    "mean": np.mean(numbers),
                    "median": np.median(numbers),
                    "std_dev": np.std(numbers),
                    "min": np.min(numbers),
                    "max": np.max(numbers)
                }
            except Exception as e:
                return f"Statistics calculation error: {str(e)}"
        
        else:
            return f"Unknown calculation type: {calculation_type}"
    
    else:
        return f"Unknown utility type: {node.data['utility_type']}"

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

    elif node.type == 'DataRetrieval':
        node.data['retrieval_type'] = st.sidebar.selectbox("Retrieval Type", ["Database", "API", "File", "Web"], key=f"retrieval_type_{node.id}")
        
        if node.data['retrieval_type'] == 'Database':
            node.data['database_type'] = st.sidebar.selectbox("Database Type", ["SQL", "NoSQL", "Vector"], key=f"database_type_{node.id}")
            node.data['query'] = st.sidebar.text_area("Query", value=node.data.get('query', ''), key=f"query_{node.id}")
            st.sidebar.info("Use {input} placeholder to insert input data into the query")
        
        elif node.data['retrieval_type'] == 'API':
            node.data['api_endpoint'] = st.sidebar.text_input("API Endpoint", value=node.data.get('api_endpoint', ''), key=f"api_endpoint_{node.id}")
            node.data['api_method'] = st.sidebar.selectbox("API Method", ["GET", "POST"], key=f"api_method_{node.id}")
            api_params_json = st.sidebar.text_area("API Parameters (JSON)", value=json.dumps(node.data.get('api_params', {}), indent=2), key=f"api_params_{node.id}")
            try:
                node.data['api_params'] = json.loads(api_params_json)
            except json.JSONDecodeError:
                st.sidebar.error("Invalid JSON for API parameters")
        
        elif node.data['retrieval_type'] == 'File':
            node.data['file_path'] = st.sidebar.text_input("File Path", value=node.data.get('file_path', ''), key=f"file_path_{node.id}")
        
        elif node.data['retrieval_type'] == 'Web':
            node.data['web_url'] = st.sidebar.text_input("URL", value=node.data.get('web_url', ''), key=f"web_url_{node.id}")
            st.sidebar.info("If URL is empty, input text will be used as URL")

    elif node.type == 'Control':
        node.data['control_type'] = st.sidebar.selectbox("Control Type", ["Conditional", "Loop", "Switch"], key=f"control_type_{node.id}")
        
        if node.data['control_type'] == 'Conditional':
            node.data['condition'] = st.sidebar.text_input("Condition", value=node.data.get('condition', "len('{input}') > 10"), key=f"condition_{node.id}")
            node.data['true_branch'] = st.sidebar.text_input("True Branch", value=node.data.get('true_branch', "Condition is true"), key=f"true_branch_{node.id}")
            node.data['false_branch'] = st.sidebar.text_input("False Branch", value=node.data.get('false_branch', "Condition is false"), key=f"false_branch_{node.id}")
            st.sidebar.info("Use {input} placeholder for input data in condition")
        
        elif node.data['control_type'] == 'Loop':
            node.data['iterations'] = st.sidebar.number_input("Iterations", min_value=1, max_value=100, value=node.data.get('iterations', 3), key=f"iterations_{node.id}")
            node.data['loop_body'] = st.sidebar.text_input("Loop Body", value=node.data.get('loop_body', "Processing {input} in iteration {i}"), key=f"loop_body_{node.id}")
            st.sidebar.info("Use {i} for iteration index and {input} for input data")
        
        elif node.data['control_type'] == 'Switch':
            cases_json = st.sidebar.text_area("Cases (JSON)", value=json.dumps(node.data.get('cases', {"case1": "Result 1", "case2": "Result 2"}), indent=2), key=f"cases_{node.id}")
            try:
                node.data['cases'] = json.loads(cases_json)
            except json.JSONDecodeError:
                st.sidebar.error("Invalid JSON for cases")
            node.data['default_case'] = st.sidebar.text_input("Default Case", value=node.data.get('default_case', "Default result"), key=f"default_case_{node.id}")

    elif node.type == 'Integration':
        node.data['integration_type'] = st.sidebar.selectbox("Integration Type", ["Email", "Webhook", "Database"], key=f"integration_type_{node.id}")
        
        if node.data['integration_type'] == 'Email':
            node.data['email_recipient'] = st.sidebar.text_input("Recipient", value=node.data.get('email_recipient', ''), key=f"email_recipient_{node.id}")
            node.data['email_subject'] = st.sidebar.text_input("Subject", value=node.data.get('email_subject', 'Workflow Notification'), key=f"email_subject_{node.id}")
            node.data['email_body'] = st.sidebar.text_area("Body", value=node.data.get('email_body', 'Workflow result: {input}'), key=f"email_body_{node.id}")
            st.sidebar.info("Use {input} placeholder to insert input data into the email body")
        
        elif node.data['integration_type'] == 'Webhook':
            node.data['webhook_url'] = st.sidebar.text_input("Webhook URL", value=node.data.get('webhook_url', ''), key=f"webhook_url_{node.id}")
            node.data['webhook_method'] = st.sidebar.selectbox("Method", ["POST", "GET"], key=f"webhook_method_{node.id}")
            webhook_data_json = st.sidebar.text_area("Webhook Data (JSON)", value=json.dumps(node.data.get('webhook_data', {"data": "{input}"}), indent=2), key=f"webhook_data_{node.id}")
            try:
                node.data['webhook_data'] = json.loads(webhook_data_json)
            except json.JSONDecodeError:
                st.sidebar.error("Invalid JSON for webhook data")
        
        elif node.data['integration_type'] == 'Database':
            node.data['database_action'] = st.sidebar.selectbox("Database Action", ["query", "insert", "update", "delete"], key=f"database_action_{node.id}")
            node.data['database_data'] = st.sidebar.text_area("Data/Query", value=node.data.get('database_data', 'SELECT * FROM table WHERE column = {input}'), key=f"database_data_{node.id}")
            st.sidebar.info("Use {input} placeholder to insert input data")

    elif node.type == 'Utility':
        node.data['utility_type'] = st.sidebar.selectbox("Utility Type", ["Format", "Transform", "Calculate"], key=f"utility_type_{node.id}")
        
        if node.data['utility_type'] == 'Format':
            node.data['format_type'] = st.sidebar.selectbox("Format Type", ["JSON", "HTML", "Markdown"], key=f"format_type_{node.id}")
        
        elif node.data['utility_type'] == 'Transform':
            node.data['transform_type'] = st.sidebar.selectbox("Transform Type", ["Uppercase", "Lowercase", "Capitalize", "Count"], key=f"transform_type_{node.id}")
        
        elif node.data['utility_type'] == 'Calculate':
            node.data['calculation_type'] = st.sidebar.selectbox("Calculation Type", ["Basic Math", "Statistics"], key=f"calculation_type_{node.id}")
            if node.data['calculation_type'] == 'Basic Math':
                node.data['formula'] = st.sidebar.text_input("Formula", value=node.data.get('formula', '{input} * 2'), key=f"formula_{node.id}")
                st.sidebar.info("Use {input} placeholder in your formula")

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
            filename = save_workflow(st.session_state['nodes'], st.session_state['edges'])
            st.success(f"Workflow saved successfully as {filename}! You can access it from the CEF Dashboard.")
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
    elif node_type == 'DataRetrieval':
        base_data.update({
            'retrieval_type': 'API',
            'api_endpoint': 'https://api.example.com/data',
            'api_method': 'GET',
            'api_params': {},
            'database_type': 'SQL',
            'query': 'SELECT * FROM table WHERE column = "{input}"',
            'file_path': '',
            'web_url': '',
        })
    elif node_type == 'Control':
        base_data.update({
            'control_type': 'Conditional',
            'condition': 'len("{input}") > 10',
            'true_branch': 'Condition is true',
            'false_branch': 'Condition is false',
            'iterations': 3,
            'loop_body': 'Processing {input} in iteration {i}',
            'cases': {
                'case1': 'Result 1',
                'case2': 'Result 2'
            },
            'default_case': 'Default result',
        })
    elif node_type == 'Integration':
        base_data.update({
            'integration_type': 'Webhook',
            'webhook_url': 'https://webhook.example.com',
            'webhook_method': 'POST',
            'webhook_data': {'data': '{input}'},
            'email_recipient': 'recipient@example.com',
            'email_subject': 'Workflow Notification',
            'email_body': 'Workflow result: {input}',
            'database_action': 'query',
            'database_data': 'SELECT * FROM table WHERE column = {input}',
        })
    elif node_type == 'Utility':
        base_data.update({
            'utility_type': 'Format',
            'format_type': 'JSON',
            'transform_type': 'Uppercase',
            'calculation_type': 'Basic Math',
            'formula': '{input} * 2',
        })
    return Node(node_id, node_type, base_data)

def save_workflow(nodes: List[Node], edges: List[Edge]) -> None:
    """Saves the current workflow to JSON files."""
    workflow_data = {
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges]
    }
    
    # Save to standard workflow.json for loading next time
    with open("workflow.json", "w") as f:
        json.dump(workflow_data, f, indent=2)
    
    # Also save with timestamp for dashboard management
    timestamp = int(time.time())
    filename = f"workflow_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(workflow_data, f, indent=2)
    
    return filename

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