# Model Management Dashboard for Ollama Workbench

## Overview

The Model Management Dashboard provides comprehensive analytics and tracking for your Ollama models. It helps you monitor usage patterns, performance metrics, resource utilization, and model metadata, enabling better decision-making about which models to use for different tasks.

## UI Fixes Update

A comprehensive set of fixes has been applied to improve the Ollama Workbench UI and functionality. These fixes address various issues with the MultiModel Chat, Tool Playground, and overall UI stability.

### How to Run With All Fixes

To run Ollama Workbench with all fixes applied, use the `run_all_fixes.sh` script:

```bash
./run_all_fixes.sh
```

### Fix Summary

#### 1. MultiModel Chat Fixes
- Fixed the "argument of type 'int' is not iterable" error for total_tokens
- Enabled multiple model selection
- Fixed embedding dimensionality mismatches

#### 2. Tool Playground Fixes
- Fixed session state conflicts with widget keys
- Fixed model selection persistence in dropdowns
- Improved error handling for models without tool support
- Added better guidance for selecting tool-compatible models

#### 3. UI Fixes
- Added stable form elements and session state management
- Fixed selectbox and multiselect widget issues
- Enhanced UI styling for better stability

#### 4. TTS Server Fixes
- Set up minimal TTS server for voice interface
- Added proper error handling for text-to-speech

For more details, run the individual fix scripts:
- `fix_multimodel_chat.py`
- `fix_embeddings.py`
- `fix_tool_playground.py`
- `fix_tool_support_warning.py`

## Features

### 1. Usage Statistics
- Track tokens generated across all models
- Monitor number of requests per model
- Analyze operation types (generate, chat, embedding, vision)
- View usage trends over time
- Compare model popularity

### 2. Performance Metrics
- Track tokens per second for each model
- Monitor response latency
- Analyze performance under different temperature settings
- Compare model performance side-by-side

### 3. Resource Utilization
- Monitor CPU usage
- Track memory consumption
- View GPU utilization (if available)
- Analyze resource usage patterns over time

### 4. Model Metadata
- View model sizes and parameter counts
- Track model capabilities (vision, tools, embedding)
- See when models were last used and modified
- Compare model capabilities side-by-side

## How to Use

1. Access the dashboard by navigating to "Maintain" > "Model Management" in the navigation menu.
2. Use the time period and model filters at the top to focus on specific data.
3. Explore different tabs to view various aspects of model usage and performance.
4. Use the refresh button to get the latest data.

## Integrating with Your Code

The dashboard automatically tracks model usage when you use the standard `call_ollama_endpoint` function. If you're using custom implementations, you can manually log usage with:

```python
from ollama_utils import log_model_stats

# After calling your model
log_model_stats(
    model_name="your_model_name",
    tokens_generated=response_tokens,  # number of tokens generated
    response_time=elapsed_time,  # time taken in seconds
    operation_type="generate"  # or "embed", "chat", "vision"
)
```

## Database

The dashboard stores data in a SQLite database (`ollama_models.db`) with these tables:

- `model_usage`: Tracks usage statistics for all models
- `model_performance`: Stores performance metrics
- `resource_utilization`: Monitors system resource usage
- `model_metadata`: Keeps track of model information

## No Data?

If you haven't used any models yet, the dashboard will show simulated data for demonstration purposes. As you use models, real data will be collected and displayed.

## Troubleshooting

- **Missing data?** Make sure models are being called through the standard Ollama utilities.
- **Performance issues?** Consider clearing older data if the database becomes very large.
- **Resource data missing?** The dashboard attempts to monitor resources when models are used, but some metrics may be unavailable depending on your system configuration.

For more help, please check the main Ollama Workbench documentation or report issues on the project repository.