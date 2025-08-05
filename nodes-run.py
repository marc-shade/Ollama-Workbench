"""
Runner for the Compound Elemental Framework (CEF) visual workflow builder.
This script provides a dashboard for managing workflows and monitoring their execution.
"""

import multiprocessing
import streamlit as st
import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Function to start the CEF workflow builder
def start_cef_app():
    import os
    os.system('python nodes.py')

# Function to load saved workflows
def load_saved_workflows():
    workflow_files = []
    try:
        for file in os.listdir():
            if file.endswith('.json') and file.startswith('workflow_'):
                workflow_files.append(file)
    except:
        pass
    return workflow_files

# Function to load workflow stats
def load_workflow_stats(workflow_file):
    try:
        with open(workflow_file, 'r') as f:
            workflow_data = json.load(f)
        
        nodes = workflow_data.get('nodes', [])
        edges = workflow_data.get('edges', [])
        
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            'filename': workflow_file,
            'nodes': len(nodes),
            'edges': len(edges),
            'node_types': node_types,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(workflow_file)).strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {
            'filename': workflow_file,
            'nodes': 0, 
            'edges': 0,
            'node_types': {},
            'last_modified': 'Unknown',
            'error': str(e)
        }

def main():
    st.set_page_config(page_title="CEF Dashboard", page_icon="✳️", layout="wide")
    
    st.title('✳️ Compound Elemental Framework Dashboard')
    st.write('Manage and monitor your AI workflows')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Workflow Builder")
        if st.button("🔄 Launch Workflow Builder", type="primary"):
            st.session_state['builder_running'] = True
            st.success("Launching workflow builder... Please wait.")
            
            # Start the builder in a background process
            if st.session_state.get('builder_process') is None or not st.session_state.get('builder_process').is_alive():
                st.session_state['builder_process'] = multiprocessing.Process(target=start_cef_app)
                st.session_state['builder_process'].start()
            
            st.info("The workflow builder is running in a separate window. Return to this dashboard to manage your workflows.")
        
        st.subheader("Saved Workflows")
        workflow_files = load_saved_workflows()
        
        if not workflow_files:
            st.info("No saved workflows found. Create a workflow using the Workflow Builder.")
        else:
            workflow_stats = [load_workflow_stats(wf) for wf in workflow_files]
            df = pd.DataFrame(workflow_stats)
            
            st.dataframe(df[['filename', 'nodes', 'edges', 'last_modified']], use_container_width=True)
            
            selected_workflow = st.selectbox("Select a workflow to view details", workflow_files)
            
            if selected_workflow:
                stats = load_workflow_stats(selected_workflow)
                
                st.subheader(f"Workflow: {selected_workflow}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    **Nodes:** {stats['nodes']}  
                    **Edges:** {stats['edges']}  
                    **Last Modified:** {stats['last_modified']}
                    """)
                
                with col_b:
                    # Display node types distribution
                    if stats['node_types']:
                        fig, ax = plt.subplots()
                        ax.bar(stats['node_types'].keys(), stats['node_types'].values())
                        ax.set_title('Node Types Distribution')
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                
                # Actions for the selected workflow
                st.subheader("Workflow Actions")
                
                col_c, col_d, col_e = st.columns(3)
                with col_c:
                    if st.button("▶️ Load in Builder", key=f"load_{selected_workflow}"):
                        # Copy the selected workflow to workflow.json
                        try:
                            with open(selected_workflow, 'r') as source:
                                with open('workflow.json', 'w') as target:
                                    target.write(source.read())
                            st.success(f"Workflow '{selected_workflow}' copied to active workflow. Launch the builder to edit it.")
                        except Exception as e:
                            st.error(f"Error loading workflow: {str(e)}")
                
                with col_d:
                    if st.button("🗑️ Delete Workflow", key=f"delete_{selected_workflow}"):
                        try:
                            os.remove(selected_workflow)
                            st.success(f"Workflow '{selected_workflow}' deleted.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting workflow: {str(e)}")
                
                with col_e:
                    if st.button("📄 Export as Script", key=f"export_{selected_workflow}"):
                        try:
                            with open(selected_workflow, 'r') as f:
                                workflow_data = json.load(f)
                            
                            # Generate a Python script from the workflow
                            script = """
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_workflow():
    results = {}
    """
                            # Add placeholders for node execution
                            for node in workflow_data.get('nodes', []):
                                node_id = node.get('id', 'unknown')
                                node_type = node.get('type', 'unknown')
                                script += f"\n    # Node {node_id}: {node_type}\n"
                                script += f"    results['{node_id}'] = 'Result of {node_type} node'\n"
                            
                            script += "\n    return results\n\n"
                            script += "if __name__ == '__main__':\n"
                            script += "    results = execute_workflow()\n"
                            script += "    print('Workflow executed successfully:')\n"
                            script += "    for node_id, result in results.items():\n"
                            script += "        print(f'Node {node_id}: {result}')\n"
                            
                            # Create a download button for the script
                            script_name = selected_workflow.replace('.json', '.py')
                            st.download_button(
                                label="Download Python Script",
                                data=script,
                                file_name=script_name,
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Error exporting workflow: {str(e)}")
    
    with col2:
        st.header("Recent Activity")
        
        # Check if any workflow files were modified recently
        recent_workflows = []
        for wf in workflow_files:
            mtime = os.path.getmtime(wf)
            if time.time() - mtime < 86400:  # Modified in the last 24 hours
                recent_workflows.append({
                    'name': wf,
                    'time': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        if recent_workflows:
            st.subheader("Recently Modified Workflows")
            for wf in sorted(recent_workflows, key=lambda x: x['time'], reverse=True):
                st.markdown(f"**{wf['name']}** - {wf['time']}")
        else:
            st.info("No recent workflow activity")
        
        st.header("Documentation")
        st.markdown("""
        ### Quick Guide
        
        1. Click **Launch Workflow Builder** to open the visual workflow editor
        2. Create your workflow by adding nodes and connections
        3. Save your workflow and return to this dashboard
        4. Select a saved workflow to view details or perform actions
        
        ### Node Types
        
        - **Input**: Starting point for data entry
        - **LLM**: Language model processing
        - **Processing**: Data transformation
        - **DataRetrieval**: Fetch data from databases or APIs
        - **Control**: Conditional logic and loops
        - **Integration**: Connect to external systems
        - **Utility**: Helper functions and transformations
        - **Output**: Final result presentation
        
        ### Tips
        
        - Use the Workflow Builder to design complex AI workflows
        - Combine multiple node types for powerful automation
        - Export workflows as Python scripts for integration with other systems
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'builder_process' not in st.session_state:
        st.session_state['builder_process'] = None
    if 'builder_running' not in st.session_state:
        st.session_state['builder_running'] = False
    
    # Run the main app
    main()