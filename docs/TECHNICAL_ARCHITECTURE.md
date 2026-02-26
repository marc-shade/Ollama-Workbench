# Technical Architecture Documentation

## Document Information
- **Version**: 1.0
- **Date**: May 22, 2025
- **Project**: Ollama Workbench Evolution
- **Status**: Active Development

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Data Architecture](#data-architecture)
4. [Security Architecture](#security-architecture)
5. [Pipeline Framework](#pipeline-framework)
6. [API Design](#api-design)
7. [Performance & Scalability](#performance--scalability)
8. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### Architecture Principles
- **Modular Design**: Loosely coupled microservices with clear boundaries
- **Local-First**: All AI processing happens locally by default
- **Extensible**: Plugin architecture supporting unlimited customization
- **Scalable**: Horizontal scaling for enterprise deployments
- **Secure**: Zero-trust security model with defense in depth

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  API Gateway                                │
│              (Authentication, Rate Limiting)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼────┐    ┌──────▼─────┐    ┌──────▼─────┐
│ Main   │    │ Pipeline   │    │ Knowledge  │
│ UI     │    │ Engine     │    │ Service    │
│Service │    │            │    │            │
└────────┘    └────────────┘    └────────────┘
    │              │                  │
    └──────────────┼──────────────────┘
                   │
        ┌──────────▼──────────┐
        │   WebSocket         │
        │   Gateway           │
        └─────────────────────┘
```

---

## Core Architecture

### 1. Service Architecture

#### Main UI Service
**Technology**: Streamlit with FastAPI backend
**Responsibilities**:
- Primary user interface
- Session management
- Real-time WebSocket communication
- Static content serving

**Key Components**:
```python
# main.py - Streamlit application entry point
# api/ - FastAPI backend routes
# components/ - Reusable UI components
# pages/ - Individual feature pages
# websocket/ - Real-time communication handlers
```

#### Pipeline Engine
**Technology**: FastAPI with Docker container orchestration
**Responsibilities**:
- Extension execution environment
- Resource isolation and management
- State persistence
- Pipeline orchestration

**Key Components**:
```python
# pipeline_server.py - FastAPI pipeline server
# containers/ - Docker container management
# execution/ - Pipeline execution engine
# state/ - State management and persistence
```

#### Knowledge Service
**Technology**: FastAPI with vector database integration
**Responsibilities**:
- Document processing and indexing
- Vector and keyword search
- RAG query processing
- Collection management

**Key Components**:
```python
# knowledge_api.py - FastAPI knowledge service
# processing/ - Document processing pipeline
# search/ - Hybrid search implementation
# collections/ - Knowledge collection management
```

### 2. Component Dependencies

```mermaid
graph TD
    A[Main UI Service] --> B[WebSocket Gateway]
    A --> C[Knowledge Service]
    A --> D[Pipeline Engine]
    
    B --> E[Redis Cache]
    C --> F[ChromaDB]
    C --> G[Elasticsearch]
    D --> H[Docker Engine]
    D --> I[State Store]
    
    E --> J[PostgreSQL]
    F --> K[Vector Storage]
    G --> L[Search Index]
    H --> M[Container Registry]
    I --> N[Redis/SQLite]
```

---

## Data Architecture

### 1. Database Schema

#### PostgreSQL (Primary Database)
```sql
-- Users and authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    oauth_provider VARCHAR(50),
    oauth_id VARCHAR(255),
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Conversations and chat history
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    model_name VARCHAR(100),
    agent_type VARCHAR(50),
    settings JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES messages(id),
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Knowledge collections
CREATE TABLE collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID REFERENCES users(id),
    access_level VARCHAR(20) DEFAULT 'private',
    settings JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    content_type VARCHAR(100),
    file_size BIGINT,
    processing_status VARCHAR(20) DEFAULT 'pending',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Pipelines and extensions
CREATE TABLE pipelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID REFERENCES users(id),
    type VARCHAR(20), -- 'tool', 'function', 'pipeline'
    config JSONB NOT NULL,
    is_public BOOLEAN DEFAULT FALSE,
    version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### ChromaDB (Vector Database)
```python
# Collection structure for embeddings
{
    "collection_name": "user_collection_uuid",
    "metadata": {
        "owner_id": "user_uuid",
        "created_at": "timestamp",
        "model": "embedding_model_name",
        "chunk_size": 1000,
        "overlap": 200
    },
    "embeddings": [...],  # Vector embeddings
    "documents": [...],   # Source text chunks
    "metadatas": [...]    # Chunk metadata
}
```

#### Redis (Cache & Session Storage)
```python
# Session data structure
session:{session_id} = {
    "user_id": "uuid",
    "conversation_id": "uuid",
    "model_settings": {...},
    "ui_state": {...},
    "expires_at": "timestamp"
}

# Real-time presence
presence:{conversation_id} = {
    "users": ["user_id_1", "user_id_2"],
    "typing": {"user_id": "timestamp"},
    "last_activity": "timestamp"
}

# Performance metrics cache
metrics:{model_name}:{time_bucket} = {
    "response_time": [float],
    "token_count": [int],
    "error_count": int,
    "usage_count": int
}
```

### 2. File Storage Architecture

#### MinIO/S3 Object Storage
```
bucket/
├── documents/
│   ├── {collection_id}/
│   │   ├── {document_id}.{ext}
│   │   └── processed/
│   │       ├── chunks.json
│   │       └── metadata.json
├── models/
│   ├── custom/
│   │   ├── {model_id}/
│   │   │   ├── config.json
│   │   │   └── weights/
├── exports/
│   ├── conversations/
│   │   └── {export_id}.json
│   └── collections/
│       └── {export_id}.zip
└── temp/
    ├── uploads/
    └── processing/
```

---

## Security Architecture

### 1. Authentication & Authorization

#### OAuth Flow Implementation
```python
# OAuth configuration
OAUTH_PROVIDERS = {
    "github": {
        "client_id": os.getenv("GITHUB_CLIENT_ID"),
        "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user"
    },
    # Additional providers...
}

# JWT token structure
{
    "sub": "user_id",
    "email": "user@example.com",
    "role": "admin|developer|user|viewer",
    "permissions": ["read", "write", "admin"],
    "exp": timestamp,
    "iat": timestamp
}
```

#### Role-Based Access Control
```python
# Permission matrix
ROLE_PERMISSIONS = {
    "admin": ["*"],  # All permissions
    "developer": [
        "conversations.read", "conversations.write",
        "collections.read", "collections.write",
        "pipelines.read", "pipelines.write", "pipelines.execute",
        "models.read", "models.manage"
    ],
    "user": [
        "conversations.read", "conversations.write",
        "collections.read", "collections.write",
        "pipelines.read", "pipelines.execute",
        "models.read"
    ],
    "viewer": [
        "conversations.read",
        "collections.read",
        "models.read"
    ]
}
```

### 2. Data Encryption

#### Encryption at Rest
- **Database**: PostgreSQL with transparent data encryption (TDE)
- **Object Storage**: MinIO with server-side encryption (SSE)
- **Vector Database**: ChromaDB with encrypted storage backend

#### Encryption in Transit
- **TLS 1.3**: All API communications
- **WebSocket Secure (WSS)**: Real-time communications
- **Certificate Management**: Automated certificate rotation

#### Secret Management
```python
# Environment-based secret management
class SecretManager:
    def __init__(self):
        self.vault = self._init_vault()
    
    def get_secret(self, key: str) -> str:
        # Priority: Environment > Vault > Default
        return (
            os.getenv(key) or
            self.vault.get(key) or
            self._get_default(key)
        )
```

---

## Pipeline Framework

### 1. Container Architecture

#### Pipeline Container Specification
```dockerfile
# Base pipeline container
FROM python:3.11-slim

# Security hardening
RUN useradd -m -u 1000 pipeline
USER pipeline

# Resource limits
LABEL max_memory="1G"
LABEL max_cpu="1.0"
LABEL network_access="restricted"

# Pipeline execution environment
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Entry point for pipeline execution
ENTRYPOINT ["python", "pipeline_runner.py"]
```

#### Container Orchestration
```python
# Pipeline execution manager
class PipelineExecutor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.resource_limits = {
            "memory": "1g",
            "cpu_period": 100000,
            "cpu_quota": 100000,  # 1 CPU core
            "network_mode": "pipeline_network"
        }
    
    async def execute_pipeline(self, pipeline_config: dict) -> dict:
        container = await self._create_container(pipeline_config)
        try:
            result = await self._run_with_timeout(container, timeout=300)
            return self._process_result(result)
        finally:
            await self._cleanup_container(container)
```

### 2. Extension Types

#### Tools (Tier 1)
```python
# Function calling tool interface
class BaseTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> dict:
        """Execute tool with given parameters"""
        pass
    
    @property
    @abstractmethod
    def schema(self) -> dict:
        """OpenAI function calling schema"""
        pass

# Example web search tool
class WebSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for current information"
        )
    
    async def execute(self, query: str, num_results: int = 5) -> dict:
        # Implementation
        return {"results": [...]}
    
    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        }
```

#### Functions (Tier 2)
```python
# UI behavior modification function
class BaseFunction:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def process_request(self, request: dict) -> dict:
        """Process incoming request and modify behavior"""
        pass
    
    @abstractmethod
    async def process_response(self, response: dict) -> dict:
        """Process outgoing response and apply modifications"""
        pass

# Example content filter function
class ContentFilterFunction(BaseFunction):
    async def process_response(self, response: dict) -> dict:
        # Apply content filtering logic
        filtered_content = self._apply_filters(response["content"])
        return {**response, "content": filtered_content}
```

#### Pipelines (Tier 3)
```python
# Complex workflow pipeline
class BasePipeline:
    def __init__(self, config: dict):
        self.config = config
        self.steps = self._parse_steps(config["steps"])
    
    async def execute(self, input_data: dict) -> dict:
        """Execute full pipeline workflow"""
        current_data = input_data
        
        for step in self.steps:
            current_data = await self._execute_step(step, current_data)
            
        return current_data
    
    async def _execute_step(self, step: dict, data: dict) -> dict:
        """Execute individual pipeline step"""
        step_type = step["type"]
        
        if step_type == "llm_call":
            return await self._execute_llm_step(step, data)
        elif step_type == "tool_call":
            return await self._execute_tool_step(step, data)
        elif step_type == "condition":
            return await self._execute_condition_step(step, data)
        else:
            raise ValueError(f"Unknown step type: {step_type}")
```

---

## API Design

### 1. REST API Endpoints

#### Core API Structure
```python
# FastAPI application structure
app = FastAPI(
    title="Ollama Workbench API",
    version="2.0.0",
    description="Comprehensive LLM platform API"
)

# API versioning
app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")

# Main API routes
@app.get("/api/v2/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
```

#### Authentication Endpoints
```python
# /api/v2/auth/
POST   /login              # Local authentication
POST   /oauth/{provider}   # OAuth authentication
POST   /refresh            # Token refresh
DELETE /logout             # Logout
GET    /profile            # User profile
PUT    /profile            # Update profile
```

#### Conversation Endpoints
```python
# /api/v2/conversations/
GET    /                   # List conversations
POST   /                   # Create conversation
GET    /{id}               # Get conversation
PUT    /{id}               # Update conversation
DELETE /{id}               # Delete conversation
POST   /{id}/messages      # Send message
GET    /{id}/messages      # Get messages
PUT    /{id}/messages/{msg_id}  # Edit message
POST   /{id}/fork          # Fork conversation
POST   /{id}/export        # Export conversation
```

#### Model Management Endpoints
```python
# /api/v2/models/
GET    /                   # List available models
POST   /pull               # Pull new model
DELETE /{name}             # Remove model
GET    /{name}/info        # Model information
POST   /{name}/test        # Test model capabilities
GET    /providers          # List providers
POST   /providers/{name}/config  # Configure provider
```

#### Knowledge Management Endpoints
```python
# /api/v2/collections/
GET    /                   # List collections
POST   /                   # Create collection
GET    /{id}               # Get collection
PUT    /{id}               # Update collection
DELETE /{id}               # Delete collection
POST   /{id}/documents     # Upload documents
DELETE /{id}/documents/{doc_id}  # Remove document
POST   /{id}/search        # Search collection
POST   /{id}/query         # RAG query
```

#### Pipeline Endpoints
```python
# /api/v2/pipelines/
GET    /                   # List pipelines
POST   /                   # Create pipeline
GET    /{id}               # Get pipeline
PUT    /{id}               # Update pipeline
DELETE /{id}               # Delete pipeline
POST   /{id}/execute       # Execute pipeline
GET    /{id}/logs          # Get execution logs
POST   /{id}/validate      # Validate pipeline config
```

### 2. WebSocket API

#### Real-Time Communication
```python
# WebSocket endpoint structure
@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    conversation_id: str,
    token: str = Query(...)
):
    # Authentication
    user = await authenticate_websocket_token(token)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    # Connection management
    await websocket.accept()
    await connection_manager.connect(websocket, conversation_id, user.id)
    
    try:
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(data, conversation_id, user.id)
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket, conversation_id, user.id)
```

#### Message Types
```python
# WebSocket message types
MESSAGE_TYPES = {
    "chat_message": {
        "type": "chat_message",
        "conversation_id": "uuid",
        "content": "string",
        "model": "string",
        "settings": {}
    },
    "typing_indicator": {
        "type": "typing_indicator",
        "conversation_id": "uuid",
        "user_id": "uuid",
        "is_typing": "boolean"
    },
    "presence_update": {
        "type": "presence_update",
        "conversation_id": "uuid",
        "users": ["uuid"],
        "action": "join|leave"
    },
    "model_response": {
        "type": "model_response",
        "conversation_id": "uuid",
        "message_id": "uuid",
        "content": "string",
        "is_complete": "boolean"
    }
}
```

---

## Performance & Scalability

### 1. Caching Strategy

#### Multi-Layer Caching
```python
# Caching architecture
class CacheManager:
    def __init__(self):
        self.l1_cache = {}  # In-memory cache
        self.l2_cache = redis.Redis()  # Redis cache
        self.l3_cache = "database"  # Database
    
    async def get(self, key: str, ttl: int = 3600):
        # L1 cache check
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # L2 cache check
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value
        
        # L3 cache (database)
        value = await self._fetch_from_database(key)
        if value:
            await self.l2_cache.setex(key, ttl, value)
            self.l1_cache[key] = value
            return value
        
        return None
```

#### Cache Invalidation
```python
# Event-driven cache invalidation
class CacheInvalidator:
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.event_handlers = {
            "conversation_updated": self._invalidate_conversation_cache,
            "model_changed": self._invalidate_model_cache,
            "collection_updated": self._invalidate_collection_cache
        }
    
    async def handle_event(self, event_type: str, data: dict):
        handler = self.event_handlers.get(event_type)
        if handler:
            await handler(data)
```

### 2. Horizontal Scaling

#### Load Balancing Configuration
```nginx
# Nginx load balancer configuration
upstream ollama_workbench {
    least_conn;
    server web1:8501 max_fails=3 fail_timeout=30s;
    server web2:8501 max_fails=3 fail_timeout=30s;
    server web3:8501 max_fails=3 fail_timeout=30s;
}

upstream pipeline_engine {
    ip_hash;  # Sticky sessions for pipeline state
    server pipeline1:8000;
    server pipeline2:8000;
    server pipeline3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://ollama_workbench;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    location /api/v2/pipelines/ {
        proxy_pass http://pipeline_engine;
    }
}
```

#### Database Scaling
```python
# Database connection pool management
class DatabaseManager:
    def __init__(self):
        self.read_pool = self._create_read_pool()
        self.write_pool = self._create_write_pool()
    
    def _create_read_pool(self):
        return create_engine(
            "postgresql://user:pass@read-replica:5432/db",
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True
        )
    
    def _create_write_pool(self):
        return create_engine(
            "postgresql://user:pass@primary:5432/db",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
    
    async def execute_read(self, query: str, params: dict = None):
        async with self.read_pool.begin() as conn:
            return await conn.execute(text(query), params)
    
    async def execute_write(self, query: str, params: dict = None):
        async with self.write_pool.begin() as conn:
            return await conn.execute(text(query), params)
```

---

## Deployment Architecture

### 1. Container Orchestration

#### Docker Compose (Development)
```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/ollama_workbench
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
      - chroma
    volumes:
      - ./:/app
      - ./data:/data

  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/ollama_workbench
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  pipeline-engine:
    build: ./pipeline
    ports:
      - "8001:8000"
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./pipelines:/pipelines

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=ollama_workbench
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8002:8000"
    volumes:
      - chroma_data:/chroma/chroma

  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - es_data:/usr/share/elasticsearch/data

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=admin
      - MINIO_ROOT_PASSWORD=password123
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  redis_data:
  chroma_data:
  es_data:
  minio_data:
```

#### Kubernetes (Production)
```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ollama-workbench

---
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: ollama-workbench
data:
  DATABASE_URL: "postgresql://postgres:password@postgres:5432/ollama_workbench"
  REDIS_URL: "redis://redis:6379"
  CHROMA_URL: "http://chroma:8000"

---
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-workbench-web
  namespace: ollama-workbench
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ollama-workbench-web
  template:
    metadata:
      labels:
        app: ollama-workbench-web
    spec:
      containers:
      - name: web
        image: ollama-workbench:latest
        ports:
        - containerPort: 8501
        envFrom:
        - configMapRef:
            name: app-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5

---
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ollama-workbench-web-service
  namespace: ollama-workbench
spec:
  selector:
    app: ollama-workbench-web
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

### 2. Monitoring & Observability

#### Prometheus Metrics
```python
# metrics/prometheus.py
from prometheus_client import Counter, Histogram, Gauge

# Application metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'websocket_connections_active',
    'Active WebSocket connections'
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference time',
    ['model_name', 'provider']
)

PIPELINE_EXECUTION_TIME = Histogram(
    'pipeline_execution_duration_seconds',
    'Pipeline execution time',
    ['pipeline_name', 'type']
)
```

#### Health Check Implementation
```python
# health/checks.py
class HealthChecker:
    def __init__(self):
        self.checks = {
            "database": self._check_database,
            "redis": self._check_redis,
            "chroma": self._check_chroma,
            "ollama": self._check_ollama,
            "disk_space": self._check_disk_space,
            "memory": self._check_memory
        }
    
    async def run_all_checks(self) -> dict:
        results = {}
        overall_status = "healthy"
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[name] = result
                if result["status"] != "healthy":
                    overall_status = "unhealthy"
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "message": str(e)
                }
                overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }
```

This technical architecture documentation provides the foundation for implementing the Ollama Workbench evolution. The modular design ensures scalability while maintaining the local-first approach that differentiates the platform.