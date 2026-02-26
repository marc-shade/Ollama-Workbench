# API Documentation - Pipeline Framework

## Document Information
- **Version**: 1.0
- **Date**: May 22, 2025
- **Project**: Ollama Workbench Pipeline Framework
- **Base URL**: `http://localhost:8001/api/v2`

## Table of Contents
1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Pipeline Management API](#pipeline-management-api)
4. [Extension API](#extension-api)
5. [Execution API](#execution-api)
6. [WebSocket API](#websocket-api)
7. [Error Handling](#error-handling)
8. [SDK Examples](#sdk-examples)

---

## API Overview

### Base Information
- **Protocol**: HTTP/1.1, HTTP/2
- **Content Type**: `application/json`
- **Authentication**: JWT Bearer Token
- **Rate Limiting**: 1000 requests/hour per user
- **Versioning**: URL path versioning (`/api/v2/`)

### Response Format
All API responses follow a consistent format:

```json
{
  "success": boolean,
  "data": object | array | null,
  "message": string,
  "timestamp": "ISO 8601 string",
  "request_id": "UUID",
  "pagination": {
    "page": number,
    "limit": number,
    "total": number,
    "has_next": boolean,
    "has_prev": boolean
  }
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

---

## Authentication

### JWT Token Authentication
All API requests require a valid JWT token in the Authorization header:

```http
Authorization: Bearer <jwt_token>
```

### Token Structure
```json
{
  "sub": "user_id",
  "email": "user@example.com",
  "role": "admin|developer|user|viewer",
  "permissions": ["pipelines.read", "pipelines.write"],
  "exp": 1640995200,
  "iat": 1640908800
}
```

### Obtaining Tokens
```http
POST /api/v2/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "expires_in": 3600,
    "token_type": "Bearer"
  }
}
```

---

## Pipeline Management API

### List Pipelines
Get a paginated list of pipelines accessible to the current user.

```http
GET /api/v2/pipelines?page=1&limit=20&type=tool&owner=me&search=web
```

**Query Parameters:**
- `page` (int, default: 1) - Page number
- `limit` (int, default: 20, max: 100) - Items per page
- `type` (string, optional) - Filter by type: `tool`, `function`, `pipeline`
- `owner` (string, optional) - Filter by owner: `me`, `public`, `{user_id}`
- `search` (string, optional) - Search in name and description
- `category` (string, optional) - Filter by category
- `sort` (string, default: `created_at`) - Sort field
- `order` (string, default: `desc`) - Sort order: `asc`, `desc`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "web_search_tool",
      "description": "Search the web for current information",
      "type": "tool",
      "category": "search",
      "owner": {
        "id": "user_id",
        "username": "john_doe",
        "email": "john@example.com"
      },
      "is_public": true,
      "version": "1.2.0",
      "tags": ["web", "search", "api"],
      "config": {...},
      "stats": {
        "downloads": 1250,
        "rating": 4.8,
        "reviews": 23
      },
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-20T14:22:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 156,
    "has_next": true,
    "has_prev": false
  }
}
```

### Get Pipeline Details
Retrieve detailed information about a specific pipeline.

```http
GET /api/v2/pipelines/{pipeline_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "web_search_tool",
    "description": "Search the web for current information using multiple search engines",
    "type": "tool",
    "category": "search",
    "owner": {
      "id": "user_id",
      "username": "john_doe",
      "email": "john@example.com",
      "avatar_url": "https://example.com/avatar.jpg"
    },
    "is_public": true,
    "version": "1.2.0",
    "tags": ["web", "search", "api"],
    "config": {
      "schema": {
        "type": "function",
        "function": {
          "name": "web_search",
          "description": "Search the web for current information",
          "parameters": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "Search query"
              },
              "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5,
                "minimum": 1,
                "maximum": 20
              }
            },
            "required": ["query"]
          }
        }
      },
      "implementation": {
        "language": "python",
        "entry_point": "main.py",
        "dependencies": ["requests", "beautifulsoup4"],
        "environment": {
          "SEARCH_API_KEY": "required"
        }
      }
    },
    "stats": {
      "downloads": 1250,
      "rating": 4.8,
      "reviews": 23,
      "usage_count": 45230
    },
    "versions": [
      {
        "version": "1.2.0",
        "created_at": "2025-01-20T14:22:00Z",
        "changelog": "Added support for multiple search engines"
      },
      {
        "version": "1.1.0",
        "created_at": "2025-01-15T10:30:00Z",
        "changelog": "Improved error handling and response formatting"
      }
    ],
    "reviews": [
      {
        "user": "alice_dev",
        "rating": 5,
        "comment": "Excellent tool, works perfectly with multiple search engines!",
        "created_at": "2025-01-18T09:15:00Z"
      }
    ],
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-20T14:22:00Z"
  }
}
```

### Create Pipeline
Create a new pipeline with the specified configuration.

```http
POST /api/v2/pipelines
Content-Type: application/json

{
  "name": "custom_web_search",
  "description": "A custom web search tool with enhanced filtering",
  "type": "tool",
  "category": "search",
  "is_public": false,
  "tags": ["web", "search", "custom"],
  "config": {
    "schema": {
      "type": "function",
      "function": {
        "name": "custom_web_search",
        "description": "Search the web with custom filters",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "Search query"
            },
            "domain_filter": {
              "type": "array",
              "items": {"type": "string"},
              "description": "Domains to include in search"
            }
          },
          "required": ["query"]
        }
      }
    },
    "implementation": {
      "language": "python",
      "entry_point": "search.py",
      "dependencies": ["requests", "urllib3"],
      "source_code": "base64_encoded_zip_file"
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "name": "custom_web_search",
    "version": "1.0.0",
    "status": "validating",
    "validation_id": "val_123456789"
  },
  "message": "Pipeline created successfully. Validation in progress."
}
```

### Update Pipeline
Update an existing pipeline configuration.

```http
PUT /api/v2/pipelines/{pipeline_id}
Content-Type: application/json

{
  "description": "Updated description with new features",
  "tags": ["web", "search", "custom", "enhanced"],
  "config": {
    // Updated configuration
  }
}
```

### Delete Pipeline
Delete a pipeline. Only the owner or admin can delete pipelines.

```http
DELETE /api/v2/pipelines/{pipeline_id}
```

**Response:**
```json
{
  "success": true,
  "message": "Pipeline deleted successfully"
}
```

---

## Extension API

### Tool Extensions (Tier 1)

#### Register Tool
Register a new tool extension for function calling.

```http
POST /api/v2/extensions/tools
Content-Type: application/json

{
  "name": "database_query",
  "description": "Execute SQL queries on connected databases",
  "schema": {
    "type": "function",
    "function": {
      "name": "db_query",
      "description": "Execute a SQL query",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "database": {"type": "string", "default": "default"}
        },
        "required": ["query"]
      }
    }
  },
  "implementation": {
    "type": "container",
    "image": "custom/db-tool:latest",
    "resource_limits": {
      "memory": "512MB",
      "cpu": "0.5"
    }
  }
}
```

#### Tool Execution
Execute a tool with specified parameters.

```http
POST /api/v2/extensions/tools/{tool_id}/execute
Content-Type: application/json

{
  "parameters": {
    "query": "SELECT * FROM users WHERE active = true",
    "database": "production"
  },
  "context": {
    "conversation_id": "conv_123",
    "user_id": "user_456"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_789",
    "result": {
      "status": "completed",
      "output": {
        "rows": 150,
        "data": [...],
        "execution_time": "0.045s"
      }
    },
    "metadata": {
      "tool_name": "database_query",
      "execution_time": 0.045,
      "resource_usage": {
        "memory": "45MB",
        "cpu": "0.1"
      }
    }
  }
}
```

### Function Extensions (Tier 2)

#### Register Function
Register a function that modifies request/response behavior.

```http
POST /api/v2/extensions/functions
Content-Type: application/json

{
  "name": "content_filter",
  "description": "Filter content based on custom rules",
  "type": "response_processor",
  "config": {
    "filter_rules": [
      {"type": "regex", "pattern": "\\b(password|secret)\\b", "action": "redact"},
      {"type": "sentiment", "threshold": -0.8, "action": "flag"}
    ],
    "processing_order": 10
  },
  "implementation": {
    "type": "inline",
    "code": "base64_encoded_python_code"
  }
}
```

#### Function Processing
Functions are automatically applied during request/response processing. Monitor function performance:

```http
GET /api/v2/extensions/functions/{function_id}/stats
```

### Pipeline Extensions (Tier 3)

#### Create Complex Pipeline
Create a multi-step workflow pipeline.

```http
POST /api/v2/extensions/pipelines
Content-Type: application/json

{
  "name": "research_workflow",
  "description": "Multi-agent research and synthesis pipeline",
  "steps": [
    {
      "id": "search_step",
      "type": "tool_call",
      "tool": "web_search",
      "parameters": {
        "query": "${input.topic}",
        "num_results": 10
      }
    },
    {
      "id": "analyze_step",
      "type": "llm_call",
      "model": "llama3",
      "prompt": "Analyze these search results: ${search_step.output}",
      "depends_on": ["search_step"]
    },
    {
      "id": "synthesize_step",
      "type": "llm_call",
      "model": "claude-3",
      "prompt": "Synthesize insights: ${analyze_step.output}",
      "depends_on": ["analyze_step"]
    }
  ],
  "output_mapping": {
    "summary": "${synthesize_step.output}",
    "sources": "${search_step.output.sources}"
  }
}
```

---

## Execution API

### Execute Pipeline
Execute a pipeline with input parameters.

```http
POST /api/v2/execution/pipelines/{pipeline_id}/execute
Content-Type: application/json

{
  "input": {
    "topic": "artificial intelligence trends 2025",
    "depth": "comprehensive"
  },
  "options": {
    "timeout": 300,
    "retry_count": 2,
    "async": true
  },
  "context": {
    "conversation_id": "conv_123",
    "user_preferences": {
      "language": "en",
      "detail_level": "high"
    }
  }
}
```

**Response (Async):**
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_550e8400",
    "status": "running",
    "pipeline_id": "pipeline_123",
    "started_at": "2025-01-22T15:30:00Z",
    "estimated_completion": "2025-01-22T15:35:00Z",
    "progress": {
      "completed_steps": 1,
      "total_steps": 3,
      "current_step": "analyze_step"
    }
  }
}
```

### Get Execution Status
Check the status of a running pipeline execution.

```http
GET /api/v2/execution/{execution_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_550e8400",
    "status": "completed",
    "pipeline_id": "pipeline_123",
    "started_at": "2025-01-22T15:30:00Z",
    "completed_at": "2025-01-22T15:34:23Z",
    "duration": 263.45,
    "progress": {
      "completed_steps": 3,
      "total_steps": 3,
      "current_step": null
    },
    "result": {
      "summary": "AI trends analysis...",
      "sources": [...]
    },
    "step_results": [
      {
        "step_id": "search_step",
        "status": "completed",
        "duration": 2.3,
        "output": {...}
      },
      {
        "step_id": "analyze_step", 
        "status": "completed",
        "duration": 45.2,
        "output": {...}
      },
      {
        "step_id": "synthesize_step",
        "status": "completed", 
        "duration": 215.95,
        "output": {...}
      }
    ],
    "resource_usage": {
      "total_memory": "1.2GB",
      "peak_cpu": "2.1",
      "network_io": "15MB"
    }
  }
}
```

### Cancel Execution
Cancel a running pipeline execution.

```http
DELETE /api/v2/execution/{execution_id}
```

### Get Execution Logs
Retrieve detailed logs for a pipeline execution.

```http
GET /api/v2/execution/{execution_id}/logs?level=info&step=analyze_step
```

**Query Parameters:**
- `level` (string, optional) - Log level: `debug`, `info`, `warn`, `error`
- `step` (string, optional) - Filter logs by step ID
- `start_time` (ISO string, optional) - Start time filter
- `end_time` (ISO string, optional) - End time filter

---

## WebSocket API

### Connection
Connect to WebSocket for real-time updates.

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/execution/{execution_id}?token={jwt_token}');
```

### Message Types

#### Execution Progress
```json
{
  "type": "execution_progress",
  "data": {
    "execution_id": "exec_550e8400",
    "progress": {
      "completed_steps": 2,
      "total_steps": 3,
      "current_step": "synthesize_step"
    },
    "step_status": {
      "step_id": "synthesize_step",
      "status": "running",
      "progress": 0.65
    }
  }
}
```

#### Step Completion
```json
{
  "type": "step_completed",
  "data": {
    "execution_id": "exec_550e8400",
    "step_id": "analyze_step",
    "status": "completed",
    "duration": 45.2,
    "output_preview": "Analysis completed successfully..."
  }
}
```

#### Execution Completion
```json
{
  "type": "execution_completed",
  "data": {
    "execution_id": "exec_550e8400",
    "status": "completed",
    "duration": 263.45,
    "result_summary": "Research workflow completed successfully"
  }
}
```

#### Error Events
```json
{
  "type": "execution_error",
  "data": {
    "execution_id": "exec_550e8400",
    "error": {
      "code": "STEP_TIMEOUT",
      "message": "Step 'analyze_step' timed out after 60 seconds",
      "step_id": "analyze_step",
      "recoverable": true
    }
  }
}
```

---

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid pipeline configuration",
    "details": {
      "field": "config.schema.function.parameters",
      "reason": "Missing required property 'type'"
    }
  },
  "timestamp": "2025-01-22T15:30:00Z",
  "request_id": "req_550e8400"
}
```

### Common Error Codes

#### Authentication Errors
- `AUTH_TOKEN_MISSING` - No authorization token provided
- `AUTH_TOKEN_INVALID` - Invalid or expired token
- `AUTH_INSUFFICIENT_PERMISSIONS` - User lacks required permissions

#### Validation Errors
- `VALIDATION_ERROR` - Request data validation failed
- `SCHEMA_INVALID` - Pipeline schema validation failed
- `DEPENDENCY_MISSING` - Required dependency not available

#### Execution Errors
- `EXECUTION_TIMEOUT` - Pipeline execution timed out
- `RESOURCE_LIMIT_EXCEEDED` - Resource limits exceeded
- `STEP_FAILED` - Individual step execution failed
- `CONTAINER_ERROR` - Container execution error

#### System Errors
- `RATE_LIMIT_EXCEEDED` - API rate limit exceeded
- `SERVICE_UNAVAILABLE` - Service temporarily unavailable
- `INTERNAL_ERROR` - Unexpected system error

---

## SDK Examples

### Python SDK

#### Installation
```bash
pip install ollama-workbench-sdk
```

#### Basic Usage
```python
from ollama_workbench import PipelineClient

# Initialize client
client = PipelineClient(
    base_url="http://localhost:8001",
    token="your_jwt_token"
)

# List pipelines
pipelines = client.pipelines.list(type="tool", limit=10)

# Get pipeline details
pipeline = client.pipelines.get("pipeline_id")

# Execute pipeline
execution = client.execute(
    pipeline_id="pipeline_id",
    input={"query": "search term"},
    async_mode=True
)

# Monitor execution
while execution.status in ["running", "queued"]:
    execution.refresh()
    print(f"Progress: {execution.progress}")
    time.sleep(2)

print(f"Result: {execution.result}")
```

#### Advanced Usage
```python
# Create a new tool
tool_config = {
    "name": "custom_tool",
    "description": "My custom tool",
    "schema": {
        "type": "function",
        "function": {
            "name": "custom_function",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }
        }
    },
    "implementation": {
        "type": "container",
        "image": "my/custom-tool:latest"
    }
}

tool = client.tools.create(tool_config)

# Execute with real-time updates
def on_progress(event):
    print(f"Step {event.step_id}: {event.progress}%")

def on_completion(event):
    print(f"Completed: {event.result}")

execution = client.execute(
    pipeline_id=tool.id,
    input={"input": "test data"},
    callbacks={
        "progress": on_progress,
        "completion": on_completion
    }
)
```

### JavaScript SDK

#### Installation
```bash
npm install @ollama-workbench/sdk
```

#### Basic Usage
```javascript
import { PipelineClient } from '@ollama-workbench/sdk';

// Initialize client
const client = new PipelineClient({
  baseURL: 'http://localhost:8001',
  token: 'your_jwt_token'
});

// List pipelines
const pipelines = await client.pipelines.list({
  type: 'tool',
  limit: 10
});

// Execute pipeline with WebSocket updates
const execution = await client.execute({
  pipelineId: 'pipeline_id',
  input: { query: 'search term' },
  onProgress: (progress) => {
    console.log(`Progress: ${progress.percentage}%`);
  },
  onStepComplete: (step) => {
    console.log(`Step ${step.id} completed`);
  },
  onComplete: (result) => {
    console.log('Execution completed:', result);
  },
  onError: (error) => {
    console.error('Execution failed:', error);
  }
});
```

### cURL Examples

#### Create and Execute Pipeline
```bash
# Create pipeline
curl -X POST http://localhost:8001/api/v2/pipelines \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_pipeline",
    "type": "tool",
    "config": {...}
  }'

# Execute pipeline
curl -X POST http://localhost:8001/api/v2/execution/pipelines/pipeline_id/execute \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {"query": "test"},
    "options": {"async": true}
  }'

# Check execution status
curl -X GET http://localhost:8001/api/v2/execution/exec_id \
  -H "Authorization: Bearer ${JWT_TOKEN}"
```

This API documentation provides comprehensive coverage of the Ollama Workbench Pipeline Framework API, enabling developers to integrate with and extend the platform effectively.