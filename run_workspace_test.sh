#!/bin/bash

# Run the collaborative workspace test app
echo "Starting Collaborative Workspace test application..."
echo "Press Ctrl+C to stop the application"

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the test app
streamlit run test_collaborative_workspace.py --server.port=8502

# Deactivate the virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi