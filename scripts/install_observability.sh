#!/bin/bash

# install_observability.sh
# Script to install and configure Opik observability for Ollama Workbench

echo "🔍 Installing Opik Observability for Ollama Workbench"
echo "======================================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider activating one:"
    echo "   source venv/bin/activate  # or your preferred virtual environment"
    echo ""
    read -p "Continue with system-wide installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

# Install Opik
echo "📦 Installing Opik..."
pip install opik>=0.2.0

if [ $? -eq 0 ]; then
    echo "✅ Opik installed successfully"
else
    echo "❌ Failed to install Opik"
    echo "You can try installing manually with: pip install opik"
    exit 1
fi

# Create data directory if it doesn't exist
echo "📁 Creating data directory..."
mkdir -p data

# Create initial configuration
echo "⚙️  Creating initial configuration..."
python3 -c "
import json
import os
from pathlib import Path

config = {
    'opik': {
        'enabled': True,
        'project_name': 'ollama-workbench',
        'api_key': None,
        'workspace': None,
        'capture_input': True,
        'capture_output': True,
        'capture_errors': True,
        'local_mode': True,
        'batch_size': 100,
        'flush_interval': 60
    },
    'privacy': {
        'hash_prompts': False,
        'truncate_responses': False,
        'max_response_length': 1000,
        'exclude_patterns': []
    },
    'performance': {
        'enable_detailed_metrics': True,
        'track_token_usage': True,
        'track_latency': True,
        'track_resource_usage': False
    },
    'alerts': {
        'enabled': True,
        'error_threshold': 0.05,
        'latency_threshold': 10.0,
        'token_usage_threshold': 10000
    },
    'retention': {
        'trace_retention_days': 30,
        'metrics_retention_days': 90,
        'log_retention_days': 7
    }
}

config_file = Path('data/observability_config.json')
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f'Configuration created at {config_file}')
"

echo ""
echo "🎉 Opik observability installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Restart your Ollama Workbench application"
echo "2. Navigate to 'Maintain' → 'Observability Dashboard'"
echo "3. Configure your Opik settings in the dashboard"
echo ""
echo "💡 Optional: Set up Opik cloud integration"
echo "   - Get an API key from https://www.comet.ml/opik"
echo "   - Add it in the Observability Dashboard configuration"
echo "   - Or set environment variable: export OPIK_API_KEY='your-key'"
echo ""
echo "📖 Documentation: Check the PRD for detailed usage instructions"
echo ""
echo "✨ Happy monitoring!"
