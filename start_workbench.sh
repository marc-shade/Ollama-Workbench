#!/bin/bash
echo "Starting Ollama Workbench..."
# Resolve the repo directory from this script's own location so the launcher
# survives volume renames (the drive has mounted as both /Volumes/FILES and
# /Volumes/FILES 1).
cd "$(dirname "$(realpath "$0")")"

# Start Ollama server if not running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Start Streamlit
echo "Starting Streamlit interface..."
# Invoke via `python -m` instead of the venv's entry-point scripts: the
# shebang lines baked into venv/bin/* point at the old mount path (and a
# path containing a space cannot work in a shebang anyway).
venv/bin/python -m streamlit run main.py
