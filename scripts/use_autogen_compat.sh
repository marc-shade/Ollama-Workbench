#!/bin/bash

# Add the current directory to PYTHONPATH to enable the compatibility layer
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Added $(pwd) to PYTHONPATH for autogen compatibility layer"

# Run the command with the modified PYTHONPATH
"$@"
