# nodes.py
import streamlit as st
import json
import os
import random
import logging
from typing import List, Dict, Union, Tuple
from ollama_utils import get_available_models, call_ollama_endpoint, load_api_keys
from openai_utils import OPENAI_MODELS, call_openai_api
from groq_utils import GROQ_MODELS, call_groq_api
from prompts import get_agent_prompt, get_metacognitive_prompt, get_voice_prompt, get_identity_prompt
import requests
import pandas as pd
import matplotlib.pyplot as plt
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_all_models():
    ollama_models = get_available_models()
    all_models = ollama_models + OPENAI_MODELS + GROQ_MODELS
    return all_models

def create_node(node_id: str, node_type: str) -> Node:
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
    elif node_type == 'DataRetrieval':
        base_data.update({
            'retrieval_type': 'Search',
            'search_query': '',
            'database_connection': '',
            'corpus_name': '',
        })
    elif node_type == 'Output':
        base_data.update({
            'output_type': 'Text',
            'output_label': 'Output:',
            'document_format': 'Text',
            'file_format': 'txt',
            'visualization_type': 'None',
        })
    elif node_type == 'Control':
        base_data.update({
            'control_type': 'Conditional',
            'condition': '',
            'true_branch': '',
            'false_branch': '',
            'loop_count': 1,
            'error_handling': 'Continue',
        })
    elif node_type == 'Integration':
        base_data.update({
            'integration_type': 'API',
            'api_endpoint': '',
            'webhook_url': '',
            'request_method': 'GET',
            'headers': {},
            'body': '',
        })
    elif node_type == 'Utility':
        base_data.update({
            'utility_type': 'Logging',
            'log_level': 'INFO',
            'notification_email': '',
            'notification_message': '',
        })
    return Node(node_id, node_type, base_data)

def execute_workflow(nodes: List[Node], edges: List[Edge]) -> Dict[str, str]:
    results = {}
    node_map = {node.id: node for node in nodes}
    api_keys = load_api_keys()

    def process_node(node_id):
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
            elif node.type == 'DataRetrieval':
                results[node_id] = handle_data_retrieval_node(node, incoming_edges, process_node)
            elif node.type == 'Output':
                results[node_id] = handle_output_node(node, incoming_edges, process_node)
            elif node.type == 'Control':
                results[node_id] = handle_control_node(node, incoming_edges, process_node)
            elif node.type == 'Integration':
                results[node_id] = handle_integration_node(node, incoming_edges, process_node)
            elif node.type == 'Utility':
                results[node_id] = handle_utility_node(node, incoming_edges, process_node)
            else:
                logger.error(f"Unknown node type: {node.type}")
                results[node_id] = f"Error: Unknown node type {node.type}"
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            results[node_id] = f"Error processing node {node_id}: {str(e)}"

        return results[node_id]

    for node in nodes:
        process_node(node.id)

    return results

def handle_input_node(node):
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

def handle_processing_node(node, incoming_edges, process_node):
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
                # This is a simplified version. In a real scenario, you'd use a proper NLP library.
                stopwords = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of'])
                input_data = ' '.join([word for word in input_data.split() if word.lower() not in stopwords])
        return input_data
    elif node.data['processing_type'] == 'Vectorization':
        # This is a placeholder. In a real scenario, you'd use a proper embedding model.
        return f"Vectorized: {input_data[:50]}..."
    else:
        return f"Unknown processing type: {node.data['processing_type']}"

def handle_llm_node(node, incoming_edges, process_node, api_keys):
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
    node.data['conversation_history'].append({"role": "user", "content": input_text})
    node.data['conversation_history'].append({"role": "assistant", "content": response})
    return response

def handle_data_retrieval_node(node, incoming_edges, process_node):
    if node.data['retrieval_type'] == 'Search':
        # Implement search logic here. This is a placeholder.
        return f"Search results for: {node.data['search_query']}"
    elif node.data['retrieval_type'] == 'RAG':
        # Implement RAG logic here. This is a placeholder.
        return "RAG results placeholder"
    elif node.data['retrieval_type'] == 'Corpus':
        # Implement corpus retrieval logic here. This is a placeholder.
        return f"Retrieved data from corpus: {node.data['corpus_name']}"
    else:
        return f"Unknown retrieval type: {node.data['retrieval_type']}"

def handle_output_node(node, incoming_edges, process_node):
    if not incoming_edges:
        return "Error: Output node requires input"
    input_data = process_node(incoming_edges[0].source)
    if node.data['output_type'] == 'Text':
        return f"{node.data['output_label']}: {input_data}"
    elif node.data['output_type'] == 'File':
        # In a real scenario, you'd save this to a file. Here we're just returning the content.
        return f"File content ({node.data['file_format']}): {input_data}"
    elif node.data['output_type'] == 'Visualization':
        # This is a placeholder. In a real scenario, you'd create an actual visualization.
        plt.figure(figsize=(10, 5))
        plt.plot([1, 2, 3, 4, 5], [1, 4, 2, 3, 5])
        plt.title(input_data[:30])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    else:
        return f"Unknown output type: {node.data['output_type']}"

def handle_control_node(node, incoming_edges, process_node):
    if node.data['control_type'] == 'Conditional':
        condition_result = eval(node.data['condition'])
        next_node = node.data['true_branch'] if condition_result else node.data['false_branch']
        return process_node(next_node)
    elif node.data['control_type'] == 'Loop':
        loop_results = []
        for _ in range(node.data['loop_count']):
            loop_results.append(process_node(node.data['true_branch']))
        return loop_results
    elif node.data['control_type'] == 'Error Handling':
        try:
            return process_node(node.data['true_branch'])
        except Exception as e:
            if node.data['error_handling'] == 'Continue':
                return f"Error occurred, but continuing: {str(e)}"
            elif node.data['error_handling'] == 'Retry':
                return process_node(node.data['true_branch'])
            else:  # Stop
                raise e
    else:
        return f"Unknown control type: {node.data['control_type']}"

def handle_integration_node(node, incoming_edges, process_node):
    if node.data['integration_type'] == 'API':
        try:
            response = requests.request(
                node.data['request_method'],
                node.data['api_endpoint'],
                headers=json.loads(node.data['headers']),
                data=node.data['body']
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"API call error: {str(e)}"
    elif node.data['integration_type'] == 'Webhook':
        try:
            response = requests.post(node.data['webhook_url'], json={"message": "Webhook triggered"})
            response.raise_for_status()
            return f"Webhook triggered: {response.text}"
        except requests.RequestException as e:
            return f"Webhook error: {str(e)}"
    else:
        return f"Unknown integration type: {node.data['integration_type']}"

def handle_utility_node(node, incoming_edges, process_node):
    if node.data['utility_type'] == 'Logging':
        log_message = process_node(incoming_edges[0].source) if incoming_edges else "No input"
        log_level = getattr(logging, node.data['log_level'])
        logger.log(log_level, f"Log from node {node.id}: {log_message}")
        return f"Logged at level {node.data['log_level']}: {log_message}"
    elif node.data['utility_type'] == 'Notification':
        notification_message = process_node(incoming_edges[0].source) if incoming_edges else "No input"
        send_email_notification(node.data['notification_email'], "Workflow Notification", notification_message)
        return f"Notification sent to {node.data['notification_email']}"
    else:
        return f"Unknown utility type: {node.data['utility_type']}"

def send_email_notification(to_email, subject, message):
    # This is a placeholder. In a real scenario, you'd use actual SMTP settings.
    logger.info(f"Sending email to {to_email}: {subject} - {message}")
    # Uncomment and fill in the following lines to send actual emails
    # smtp_server = "smtp.gmail.com"
    # port = 587
    # sender_email = "your_email@gmail.com"
    # password = "your_password"
    # 
    # msg = MIMEMultipart()
    # msg["From"] = sender_email
    # msg["To"] = to_email
    # msg["Subject"] = subject
    # msg.attach(MIMEText(message, "plain"))
    # 
    # try:
    #     server = smtplib.SMTP(smtp_server, port)
    #     server.starttls()
    #     server.login(sender_email, password)
    #     server.send_message(msg)
    # except Exception as e:
    #     logger.error(f"Error sending email: {str(e)}")
    # finally:
    #     server.quit()

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
    
    if node.data['conversation_history']:
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in node.data['conversation_history'][-5:]])
        prompt_parts.append(f"Recent Conversation:\n{history}")
    
    return "\n\n".join(prompt_parts)

def validate_workflow(nodes: List[Node], edges: List[Edge]) -> Dict[str, Union[bool, str]]:
    if not any(node.type == 'Input' for node in nodes):
        return {'valid': False, 'error': 'Workflow must have at least one Input node.'}

    if not any(node.type == 'Output' for node in nodes):
        return {'valid': False, 'error': 'Workflow must have at least one Output node.'}

    node_ids = set(node.id for node in nodes)
    connected_nodes = set()
    for edge in edges:
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

def render_node_settings(node: Node):
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
        node.data['processing_type'] = st.sidebar.selectbox("Processing Type", ["Preprocessing", "Vectorization"], key=f"processing_type_{node.id}")
        if node.data['processing_type'] == 'Preprocessing':
            node.data['preprocessing_steps'] = st.sidebar.multiselect("Preprocessing Steps", ["Tokenization", "Lowercasing", "Remove Punctuation", "Remove Stopwords"], key=f"preprocessing_steps_{node.id}")
        elif node.data['processing_type'] == 'Vectorization':
            node.data['vectorization_model'] = st.sidebar.selectbox("Vectorization Model", ["Word2Vec", "GloVe", "FastText"], key=f"vectorization_model_{node.id}")
    elif node.type == 'LLM':
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
        node.data['fine_tuning'] = st.sidebar.checkbox("Enable Fine-tuning", value=node.data['fine_tuning'], key=f"fine_tuning_{node.id}")
    elif node.type == 'DataRetrieval':
        node.data['retrieval_type'] = st.sidebar.selectbox("Retrieval Type", ["Search", "RAG", "Corpus"], key=f"retrieval_type_{node.id}")
        if node.data['retrieval_type'] == 'Search':
            node.data['search_query'] = st.sidebar.text_input("Search Query", value=node.data['search_query'], key=f"search_query_{node.id}")
        elif node.data['retrieval_type'] == 'RAG':
            node.data['database_connection'] = st.sidebar.text_input("Database Connection String", value=node.data['database_connection'], key=f"database_connection_{node.id}")
        elif node.data['retrieval_type'] == 'Corpus':
            node.data['corpus_name'] = st.sidebar.text_input("Corpus Name", value=node.data['corpus_name'], key=f"corpus_name_{node.id}")
    elif node.type == 'Output':
        node.data['output_type'] = st.sidebar.selectbox("Output Type", ["Text", "File", "Visualization"], key=f"output_type_{node.id}")
        node.data['output_label'] = st.sidebar.text_input("Output Label", value=node.data['output_label'], key=f"output_label_{node.id}")
        if node.data['output_type'] == 'Text':
            node.data['document_format'] = st.sidebar.selectbox("Document Format", ["Text", "Markdown", "HTML"], index=["Text", "Markdown", "HTML"].index(node.data['document_format']), key=f"document_format_{node.id}")
        elif node.data['output_type'] == 'File':
            node.data['file_format'] = st.sidebar.selectbox("File Format", ["txt", "csv", "json"], key=f"file_format_{node.id}")
        elif node.data['output_type'] == 'Visualization':
            node.data['visualization_type'] = st.sidebar.selectbox("Visualization Type", ["Bar Chart", "Line Chart", "Scatter Plot"], key=f"visualization_type_{node.id}")
    elif node.type == 'Control':
        node.data['control_type'] = st.sidebar.selectbox("Control Type", ["Conditional", "Loop", "Error Handling"], key=f"control_type_{node.id}")
        if node.data['control_type'] == 'Conditional':
            node.data['condition'] = st.sidebar.text_input("Condition", value=node.data['condition'], key=f"condition_{node.id}")
            node.data['true_branch'] = st.sidebar.text_input("True Branch Node ID", value=node.data['true_branch'], key=f"true_branch_{node.id}")
            node.data['false_branch'] = st.sidebar.text_input("False Branch Node ID", value=node.data['false_branch'], key=f"false_branch_{node.id}")
        elif node.data['control_type'] == 'Loop':
            node.data['loop_count'] = st.sidebar.number_input("Loop Count", min_value=1, value=node.data['loop_count'], key=f"loop_count_{node.id}")
        elif node.data['control_type'] == 'Error Handling':
            node.data['error_handling'] = st.sidebar.selectbox("Error Handling", ["Continue", "Retry", "Stop"], key=f"error_handling_{node.id}")
    elif node.type == 'Integration':
        node.data['integration_type'] = st.sidebar.selectbox("Integration Type", ["API", "Webhook"], key=f"integration_type_{node.id}")
        if node.data['integration_type'] == 'API':
            node.data['api_endpoint'] = st.sidebar.text_input("API Endpoint", value=node.data['api_endpoint'], key=f"api_endpoint_{node.id}")
            node.data['request_method'] = st.sidebar.selectbox("Request Method", ["GET", "POST", "PUT", "DELETE"], key=f"request_method_{node.id}")
            node.data['headers'] = st.sidebar.text_area("Headers (JSON)", value=json.dumps(node.data['headers']), key=f"headers_{node.id}")
            node.data['body'] = st.sidebar.text_area("Request Body", value=node.data['body'], key=f"body_{node.id}")
        elif node.data['integration_type'] == 'Webhook':
            node.data['webhook_url'] = st.sidebar.text_input("Webhook URL", value=node.data['webhook_url'], key=f"webhook_url_{node.id}")
    elif node.type == 'Utility':
        node.data['utility_type'] = st.sidebar.selectbox("Utility Type", ["Logging", "Notification"], key=f"utility_type_{node.id}")
        if node.data['utility_type'] == 'Logging':
            node.data['log_level'] = st.sidebar.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], key=f"log_level_{node.id}")
        elif node.data['utility_type'] == 'Notification':
            node.data['notification_email'] = st.sidebar.text_input("Notification Email", value=node.data['notification_email'], key=f"notification_email_{node.id}")
            node.data['notification_message'] = st.sidebar.text_area("Notification Message", value=node.data['notification_message'], key=f"notification_message_{node.id}")

    if st.sidebar.button("Update Node", key=f"update_node_{node.id}"):
        st.success(f"Node {node.id} updated successfully!")
        st.rerun()

def render_workflow_canvas(nodes: List[Node], edges: List[Edge]):
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
                if node.type == 'Input':
                    st.write(f"**Input Type:** {node.data['input_type']}")
                elif node.type == 'Processing':
                    st.write(f"**Processing Type:** {node.data['processing_type']}")
                elif node.type == 'LLM':
                    st.write(f"**Model:** {node.data['model_name']}")
                elif node.type == 'DataRetrieval':
                    st.write(f"**Retrieval Type:** {node.data['retrieval_type']}")
                elif node.type == 'Output':
                    st.write(f"**Output Type:** {node.data['output_type']}")
                elif node.type == 'Control':
                    st.write(f"**Control Type:** {node.data['control_type']}")
                elif node.type == 'Integration':
                    st.write(f"**Integration Type:** {node.data['integration_type']}")
                elif node.type == 'Utility':
                    st.write(f"**Utility Type:** {node.data['utility_type']}")
                st.markdown("</div>", unsafe_allow_html=True)

    # Render edges
    st.subheader("🔀 Connections")
    for edge in edges:
        st.markdown(f"<div style='text-align: center; font-weight: bold;'>Node {edge.source} ➡️ Node {edge.target}</div>", unsafe_allow_html=True)

def nodes_interface():
    st.title("🧩 Nodes")

    # Add custom CSS for danger buttons
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

    with st.sidebar:
        st.subheader("🛠️ Workflow Actions")
        
        st.subheader("➕ Add New Node")
        node_type = st.selectbox("Select node type", ["Input", "Processing", "LLM", "DataRetrieval", "Output", "Control", "Integration", "Utility"])
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
                    if node.data['output_type'] == 'Text':
                        if node.data['document_format'] == 'Markdown':
                            st.markdown(result)
                        elif node.data['document_format'] == 'HTML':
                            st.components.v1.html(result, height=300)
                        else:
                            st.text(result)
                    elif node.data['output_type'] == 'File':
                        st.download_button(
                            label="Download Output File",
                            data=result,
                            file_name=f"output.{node.data['file_format']}",
                            mime=f"text/{node.data['file_format']}"
                        )
                    elif node.data['output_type'] == 'Visualization':
                        st.image(result)
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
            for key, value in node.data.items():
                if key != 'content':
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")

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
        1. **Add Nodes**: Use the sidebar to add various types of nodes (Input, Processing, LLM, DataRetrieval, Output, Control, Integration, Utility).
        2. **Connect Nodes**: Add edges to connect nodes in the desired order.
        3. **Configure Nodes**: Click on a node in the canvas to configure its settings in the sidebar.
        4. **Execute Workflow**: Click the 'Execute Workflow' button to run your workflow.
        5. **Save/Load**: Save your workflow for later use or load a previously saved workflow.
        6. **Validate**: Use the 'Validate Workflow' button to check if your workflow is properly constructed.
        7. **Remove**: Remove nodes or edges using the removal section at the bottom.

        ### Node Types:
        - **Input**: For entering text data, uploading files, or fetching data from APIs.
        - **Processing**: For preprocessing data or vectorizing text.
        - **LLM**: Language Model nodes for generating text, answering questions, or processing inputs.
        - **DataRetrieval**: For searching, implementing RAG, or accessing local corpus.
        - **Output**: For displaying text, saving files, or creating visualizations.
        - **Control**: For implementing conditional logic, loops, or error handling.
        - **Integration**: For making API calls or triggering webhooks.
        - **Utility**: For logging or sending notifications.

        For more detailed instructions, please refer to the documentation.
        """)

def save_workflow(nodes: List[Node], edges: List[Edge]):
    workflow_data = {
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges]
    }
    with open("workflow.json", "w") as f:
        json.dump(workflow_data, f, indent=2)

def load_workflow() -> Tuple[List[Node], List[Edge]]:
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

# Add a function to export the workflow as a standalone Python script
def export_workflow_as_script(nodes: List[Node], edges: List[Edge]) -> str:
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
        elif node.type == 'DataRetrieval':
            script += f"    # Add data retrieval logic here\n"
            script += f"    results['{node.id}'] = 'Retrieved data'  # Placeholder\n"
        elif node.type == 'Output':
            script += f"    input_data = results['{edges[0].source}']\n"
            script += f"    print(f'Output: {{input_data}}')\n"
        elif node.type == 'Control':
            script += f"    # Add control logic here\n"
        elif node.type == 'Integration':
            script += f"    # Add integration logic here\n"
        elif node.type == 'Utility':
            script += f"    # Add utility logic here\n"

    script += "\n    return results\n\n"
    script += "if __name__ == '__main__':\n"
    script += "    execute_workflow()\n"

    return script

# Main execution
if __name__ == "__main__":
    nodes_interface()