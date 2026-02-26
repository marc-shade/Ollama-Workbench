# Contributing to Ollama Workbench

## Welcome Contributors!

Thank you for your interest in contributing to Ollama Workbench! This guide will help you get started with contributing to our comprehensive local LLM platform. Whether you're fixing bugs, adding features, improving documentation, or creating extensions, your contributions are valuable to our community.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Environment Setup](#development-environment-setup)
3. [Contribution Types](#contribution-types)
4. [Development Workflow](#development-workflow)
5. [Code Standards](#code-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation Guidelines](#documentation-guidelines)
8. [Extension Development](#extension-development)
9. [Community Guidelines](#community-guidelines)

---

## Getting Started

### Prerequisites
- **Python 3.11+**: Core language for the platform
- **Node.js 18+**: For frontend development and tooling
- **Docker**: For containerized development and pipeline execution
- **Git**: Version control
- **Ollama**: Local LLM runtime (installed automatically via scripts)

### Quick Start
1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/Ollama-Workbench.git
   cd Ollama-Workbench
   ```

2. **Set Up Development Environment**
   ```bash
   # Run the setup script
   ./setup_development.sh
   
   # Or manually:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install Ollama and Dependencies**
   ```bash
   ./install_ollama.sh
   ./setup_workbench.sh
   ```

4. **Start Development Environment**
   ```bash
   ./start_workbench.sh
   ```

5. **Verify Installation**
   ```bash
   python -m pytest tests/ -v
   ```

---

## Development Environment Setup

### Recommended IDE Setup

#### VS Code (Recommended)
Install these extensions:
- Python
- Pylance
- Black Formatter
- Flake8
- Docker
- GitLens
- Thunder Client (for API testing)

**VS Code Settings** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "files.associations": {
    "*.md": "markdown"
  }
}
```

#### PyCharm
1. Open the project directory
2. Configure Python interpreter to use `./venv/bin/python`
3. Enable pytest as the test runner
4. Configure Black as the code formatter

### Environment Configuration

#### Required Environment Variables
Create a `.env` file in the project root:
```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/ollama_workbench
TEST_DATABASE_URL=postgresql://postgres:password@localhost:5432/ollama_workbench_test

# Redis Configuration
REDIS_URL=redis://localhost:6379

# API Keys (optional for development)
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
MISTRAL_API_KEY=your_mistral_key_here

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

#### Docker Development Environment
For a fully containerized development experience:

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Rebuild services
docker-compose -f docker-compose.dev.yml build --no-cache
```

---

## Contribution Types

### 🐛 Bug Fixes
- **Scope**: Critical bugs, performance issues, security vulnerabilities
- **Process**: Create issue → Fork → Fix → Test → Submit PR
- **Priority**: Security > Critical > High > Medium > Low

### ✨ Feature Development
- **Scope**: New features from the roadmap or community-requested features
- **Process**: Discuss in issue → Design document → Implementation → Testing → Documentation
- **Categories**: UI/UX, Backend APIs, AI/ML features, Extensions

### 📚 Documentation
- **Scope**: API docs, user guides, tutorials, code comments
- **Process**: Identify gaps → Write/Update → Review → Publish
- **Types**: Technical docs, user guides, API references, tutorials

### 🔧 Infrastructure & DevOps
- **Scope**: CI/CD, deployment, monitoring, performance optimization
- **Process**: Proposal → Implementation → Testing → Deployment
- **Areas**: Docker, Kubernetes, monitoring, security

### 🧩 Extension Development
- **Scope**: Tools, functions, pipelines for the marketplace
- **Process**: Design → Implement → Test → Document → Submit
- **Types**: Tools (Tier 1), Functions (Tier 2), Pipelines (Tier 3)

---

## Development Workflow

### 1. Issue Creation
Before starting work, create or find an existing issue:

**Bug Reports:**
```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Expected vs actual behavior

## Environment
- OS: [macOS/Windows/Linux]
- Python version: [3.11.x]
- Ollama version: [x.x.x]
- Browser: [if applicable]

## Additional Context
Screenshots, logs, or additional information
```

**Feature Requests:**
```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? Who will benefit?

## Proposed Implementation
High-level approach (if known)

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3
```

### 2. Branch Strategy
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Create bugfix branch
git checkout -b bugfix/issue-number-description

# Create documentation branch
git checkout -b docs/documentation-improvement
```

**Branch Naming Conventions:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test improvements
- `chore/` - Maintenance tasks

### 3. Development Process

#### Pre-development
```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create and switch to feature branch
git checkout -b feature/your-feature

# Install development dependencies
pip install -r requirements-dev.txt
```

#### During Development
```bash
# Run tests frequently
python -m pytest tests/ -v

# Check code quality
flake8 .
black --check .
mypy .

# Run specific tests
python -m pytest tests/test_your_module.py -v
```

#### Pre-commit Checks
```bash
# Format code
black .

# Sort imports
isort .

# Run linting
flake8 .

# Type checking
mypy .

# Run full test suite
python -m pytest tests/ --cov=. --cov-report=html
```

### 4. Commit Guidelines

#### Commit Message Format
```
type(scope): description

Longer description if needed

Closes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(chat): add conversation branching functionality

- Implement conversation fork and merge
- Add UI controls for branch management
- Update database schema for conversation trees

Closes #123

fix(pipeline): resolve container timeout issues

- Increase default timeout from 60 to 300 seconds
- Add configurable timeout per pipeline
- Improve error handling for timeout scenarios

Closes #456
```

### 5. Pull Request Process

#### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] New tests added (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Closes #issue_number
```

#### PR Review Process
1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: At least one maintainer review required
3. **Manual Testing**: Reviewer tests functionality
4. **Documentation Review**: Check for documentation updates
5. **Approval**: PR approved and merged

---

## Code Standards

### Python Code Style

#### Formatting with Black
```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .

# Format specific file
black path/to/file.py
```

#### Linting with Flake8
Configuration in `setup.cfg`:
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = 
    .git,
    __pycache__,
    venv,
    build,
    dist
```

#### Type Hints with MyPy
```python
from typing import Dict, List, Optional, Union, Any
import asyncio

async def process_pipeline(
    pipeline_id: str,
    input_data: Dict[str, Any],
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Process a pipeline with the given input data.
    
    Args:
        pipeline_id: Unique identifier for the pipeline
        input_data: Input parameters for pipeline execution
        timeout: Optional timeout in seconds
        
    Returns:
        Dictionary containing execution results
        
    Raises:
        PipelineNotFoundError: If pipeline doesn't exist
        ExecutionTimeoutError: If execution exceeds timeout
    """
    # Implementation here
    pass
```

### Code Organization

#### Module Structure
```
project/
├── __init__.py
├── main.py              # Entry point
├── api/                 # API routes and handlers
│   ├── __init__.py
│   ├── auth.py
│   ├── pipelines.py
│   └── models.py
├── core/                # Core business logic
│   ├── __init__.py
│   ├── pipeline.py
│   ├── execution.py
│   └── models.py
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── database.py
│   ├── logging.py
│   └── helpers.py
└── tests/               # Test files
    ├── __init__.py
    ├── test_pipeline.py
    └── conftest.py
```

#### Import Organization
```python
# Standard library imports
import asyncio
import json
import logging
from typing import Dict, List, Optional

# Third-party imports
import streamlit as st
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

# Local imports
from core.pipeline import PipelineEngine
from utils.database import get_database_connection
from utils.logging import setup_logger
```

### Documentation Standards

#### Docstring Format (Google Style)
```python
class PipelineEngine:
    """Engine for executing AI pipelines.
    
    This class manages the execution of complex AI workflows including
    tool calling, multi-model orchestration, and state management.
    
    Attributes:
        container_manager: Manages Docker containers for pipeline execution
        state_store: Persistent storage for pipeline state
        
    Example:
        >>> engine = PipelineEngine()
        >>> result = await engine.execute(pipeline_id, input_data)
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the pipeline engine.
        
        Args:
            config: Configuration dictionary containing engine settings
            
        Raises:
            ConfigurationError: If required configuration is missing
        """
        pass
    
    async def execute(
        self,
        pipeline_id: str,
        input_data: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a pipeline with the given input.
        
        Args:
            pipeline_id: Unique identifier for the pipeline to execute
            input_data: Input parameters for the pipeline
            timeout: Optional timeout in seconds (default: 300)
            
        Returns:
            Dictionary containing:
                - result: The pipeline execution result
                - metadata: Execution metadata (timing, resources, etc.)
                - logs: Execution logs
                
        Raises:
            PipelineNotFoundError: If the pipeline doesn't exist
            ExecutionTimeoutError: If execution exceeds timeout
            ValidationError: If input_data is invalid
            
        Example:
            >>> result = await engine.execute(
            ...     "web-search-pipeline",
            ...     {"query": "AI trends 2025"}
            ... )
            >>> print(result["result"])
        """
        pass
```

---

## Testing Guidelines

### Test Structure

#### Test Organization
```
tests/
├── conftest.py          # Pytest configuration and fixtures
├── unit/                # Unit tests
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/         # Integration tests
│   ├── test_api.py
│   ├── test_database.py
│   └── test_pipeline_execution.py
├── e2e/                 # End-to-end tests
│   ├── test_workflows.py
│   └── test_ui.py
└── fixtures/            # Test data and fixtures
    ├── sample_pipelines.json
    └── test_documents/
```

#### Test Fixtures (`conftest.py`)
```python
import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import create_test_database, get_database_session
from core.pipeline import PipelineEngine

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_database():
    """Create test database for the session."""
    db = await create_test_database()
    yield db
    await db.close()

@pytest.fixture
async def db_session(test_database) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for each test."""
    async with get_database_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def pipeline_engine(db_session):
    """Create a pipeline engine instance for testing."""
    return PipelineEngine(
        database=db_session,
        config={"mode": "test"}
    )

@pytest.fixture
def sample_pipeline():
    """Sample pipeline configuration for testing."""
    return {
        "name": "test_pipeline",
        "type": "tool",
        "config": {
            "schema": {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
```

### Unit Tests

#### Testing Functions
```python
import pytest
from unittest.mock import AsyncMock, patch
from core.pipeline import PipelineEngine, PipelineNotFoundError

class TestPipelineEngine:
    """Test cases for PipelineEngine class."""
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_success(self, pipeline_engine, sample_pipeline):
        """Test successful pipeline execution."""
        # Arrange
        pipeline_id = "test_pipeline_id"
        input_data = {"input": "test data"}
        expected_result = {"output": "processed data"}
        
        with patch.object(pipeline_engine, '_load_pipeline') as mock_load:
            mock_load.return_value = sample_pipeline
            with patch.object(pipeline_engine, '_execute_steps') as mock_execute:
                mock_execute.return_value = expected_result
                
                # Act
                result = await pipeline_engine.execute(pipeline_id, input_data)
                
                # Assert
                assert result["result"] == expected_result
                assert "metadata" in result
                assert "logs" in result
                mock_load.assert_called_once_with(pipeline_id)
                mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_not_found(self, pipeline_engine):
        """Test pipeline execution with non-existent pipeline."""
        # Arrange
        pipeline_id = "non_existent_pipeline"
        input_data = {"input": "test data"}
        
        with patch.object(pipeline_engine, '_load_pipeline') as mock_load:
            mock_load.side_effect = PipelineNotFoundError(f"Pipeline {pipeline_id} not found")
            
            # Act & Assert
            with pytest.raises(PipelineNotFoundError):
                await pipeline_engine.execute(pipeline_id, input_data)
    
    @pytest.mark.parametrize("timeout,expected_timeout", [
        (None, 300),  # Default timeout
        (60, 60),     # Custom timeout
        (0, 300),     # Invalid timeout uses default
    ])
    @pytest.mark.asyncio
    async def test_execute_pipeline_timeout_handling(
        self, 
        pipeline_engine, 
        sample_pipeline, 
        timeout, 
        expected_timeout
    ):
        """Test timeout parameter handling."""
        pipeline_id = "test_pipeline_id"
        input_data = {"input": "test data"}
        
        with patch.object(pipeline_engine, '_load_pipeline') as mock_load:
            mock_load.return_value = sample_pipeline
            with patch.object(pipeline_engine, '_execute_steps') as mock_execute:
                mock_execute.return_value = {"output": "result"}
                
                await pipeline_engine.execute(pipeline_id, input_data, timeout)
                
                # Verify timeout was set correctly
                call_args = mock_execute.call_args
                assert call_args[1]['timeout'] == expected_timeout
```

### Integration Tests

#### API Testing
```python
import pytest
from httpx import AsyncClient
from fastapi import status

from main import app

class TestPipelineAPI:
    """Integration tests for Pipeline API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_pipeline(self, auth_headers, sample_pipeline):
        """Test pipeline creation endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v2/pipelines",
                json=sample_pipeline,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["success"] is True
            assert "id" in data["data"]
            assert data["data"]["name"] == sample_pipeline["name"]
    
    @pytest.mark.asyncio
    async def test_execute_pipeline(self, auth_headers, created_pipeline):
        """Test pipeline execution endpoint."""
        execution_data = {
            "input": {"query": "test query"},
            "options": {"async": True}
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                f"/api/v2/execution/pipelines/{created_pipeline.id}/execute",
                json=execution_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert "execution_id" in data["data"]
            assert data["data"]["status"] in ["running", "queued"]
```

### End-to-End Tests

#### Workflow Testing
```python
import pytest
from playwright.async_api import async_playwright

class TestWorkflows:
    """End-to-end tests for complete workflows."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_pipeline_workflow(self):
        """Test complete pipeline creation and execution workflow."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            try:
                # Navigate to application
                await page.goto("http://localhost:8501")
                
                # Login
                await page.fill("input[data-testid='email']", "test@example.com")
                await page.fill("input[data-testid='password']", "password")
                await page.click("button[data-testid='login']")
                
                # Navigate to pipeline creation
                await page.click("text=Pipelines")
                await page.click("button[data-testid='create-pipeline']")
                
                # Fill pipeline form
                await page.fill("input[data-testid='pipeline-name']", "E2E Test Pipeline")
                await page.fill("textarea[data-testid='pipeline-description']", "Test pipeline")
                
                # Create pipeline
                await page.click("button[data-testid='save-pipeline']")
                
                # Verify creation
                await page.wait_for_selector("text=Pipeline created successfully")
                
                # Execute pipeline
                await page.click("button[data-testid='execute-pipeline']")
                await page.fill("textarea[data-testid='input-data']", '{"test": "data"}')
                await page.click("button[data-testid='run-execution']")
                
                # Wait for and verify results
                await page.wait_for_selector("text=Execution completed", timeout=30000)
                execution_result = await page.text_content("[data-testid='execution-result']")
                assert execution_result is not None
                
            finally:
                await browser.close()
```

### Test Running Commands

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v                    # Unit tests only
python -m pytest tests/integration/ -v             # Integration tests
python -m pytest -m e2e                           # End-to-end tests

# Run tests with specific markers
python -m pytest -m "not slow"                    # Skip slow tests
python -m pytest -m "api"                         # API tests only

# Run tests in parallel
python -m pytest -n auto                          # Use all CPU cores
python -m pytest -n 4                             # Use 4 processes

# Run with debugging
python -m pytest --pdb                            # Drop to debugger on failure
python -m pytest -s                               # Don't capture output
```

---

## Documentation Guidelines

### Documentation Types

#### Code Documentation
- **Docstrings**: All public functions, classes, and modules
- **Inline Comments**: Complex logic and algorithms
- **Type Hints**: All function signatures
- **README Files**: For each major module

#### User Documentation
- **User Guides**: Step-by-step instructions
- **Tutorials**: Learning-oriented content
- **API Reference**: Complete API documentation
- **FAQ**: Common questions and solutions

#### Developer Documentation
- **Architecture Docs**: System design and patterns
- **Contribution Guides**: This document
- **Deployment Guides**: Setup and deployment instructions
- **Troubleshooting**: Common issues and solutions

### Documentation Standards

#### Markdown Formatting
```markdown
# Main Title (H1)

## Section Title (H2)

### Subsection (H3)

#### Sub-subsection (H4)

**Bold text** for emphasis
*Italic text* for slight emphasis
`inline code` for code elements
~~strikethrough~~ for deprecated content

> Blockquotes for important notes

- Bullet lists for items
- Use consistent bullet style

1. Numbered lists for procedures
2. Use for step-by-step instructions

[Link text](URL) for external links
[Internal link](#anchor) for document navigation

```python
# Code blocks with language specification
def example_function():
    return "hello world"
```

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

#### API Documentation Format
```python
"""
Module: pipeline_api.py
Description: API endpoints for pipeline management

This module provides REST API endpoints for creating, managing,
and executing AI pipelines within the Ollama Workbench platform.

Example:
    Basic usage of the pipeline API:
    
    >>> import httpx
    >>> client = httpx.Client(base_url="http://localhost:8001")
    >>> response = client.get("/api/v2/pipelines")
    >>> pipelines = response.json()
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends

router = APIRouter(prefix="/api/v2/pipelines", tags=["pipelines"])

@router.get("/", response_model=List[Pipeline])
async def list_pipelines(
    page: int = 1,
    limit: int = 20,
    type_filter: Optional[str] = None,
    user: User = Depends(get_current_user)
) -> List[Pipeline]:
    """List pipelines accessible to the current user.
    
    Retrieves a paginated list of pipelines that the authenticated user
    has permission to access. Results can be filtered by pipeline type.
    
    Args:
        page: Page number for pagination (starts at 1)
        limit: Maximum number of pipelines per page (max: 100)
        type_filter: Optional filter by pipeline type ('tool', 'function', 'pipeline')
        user: Authenticated user (injected by dependency)
        
    Returns:
        List of Pipeline objects matching the criteria
        
    Raises:
        HTTPException: 400 if pagination parameters are invalid
        HTTPException: 403 if user lacks pipeline access permissions
        
    Example:
        GET /api/v2/pipelines?page=1&limit=10&type=tool
        
        Response:
        {
            "success": true,
            "data": [
                {
                    "id": "pipeline-uuid",
                    "name": "web_search",
                    "type": "tool",
                    ...
                }
            ],
            "pagination": {...}
        }
    """
    # Implementation here
    pass
```

### Documentation Workflow

#### Creating Documentation
1. **Identify Need**: Gap in existing documentation
2. **Plan Structure**: Outline and organization
3. **Write Content**: Clear, concise, accurate
4. **Review**: Self-review for clarity and accuracy
5. **Get Feedback**: Peer review and user testing
6. **Iterate**: Improve based on feedback
7. **Publish**: Merge and deploy

#### Updating Documentation
1. **Code Changes**: Update docs when code changes
2. **User Feedback**: Address user-reported issues
3. **Regular Review**: Periodic documentation audits
4. **Version Control**: Track documentation changes

---

## Extension Development

### Extension Types Overview

#### Tools (Tier 1) - Function Calling Extensions
Tools are the simplest extension type, providing specific functionality that LLMs can call during conversations.

**Use Cases:**
- Web search and data retrieval
- Database queries and operations
- File system operations
- API integrations
- Mathematical calculations

**Example Tool Structure:**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import requests

class BaseTool(ABC):
    """Base class for all tool extensions."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """OpenAI function calling schema."""
        pass

class WebSearchTool(BaseTool):
    """Tool for searching the web using multiple search engines."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for current information"
        )
        self.api_key = os.getenv("SEARCH_API_KEY")
    
    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute web search with the given query.
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-20)
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            # Implement search logic
            results = await self._perform_search(query, num_results)
            
            return {
                "success": True,
                "results": results,
                "query": query,
                "count": len(results),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    @property
    def schema(self) -> Dict[str, Any]:
        """OpenAI function calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
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
        }
    
    async def _perform_search(self, query: str, num_results: int) -> List[Dict]:
        """Internal method to perform the actual search."""
        # Implementation details
        pass
```

**Tool Registration:**
```python
# tools/my_tool.py
from tools.base import BaseTool

class MyCustomTool(BaseTool):
    # Implementation here
    pass

# Register tool
def register_tool():
    return MyCustomTool()

# Tool metadata
TOOL_METADATA = {
    "name": "my_custom_tool",
    "version": "1.0.0",
    "author": "Your Name",
    "description": "Description of what this tool does",
    "category": "utility",
    "tags": ["custom", "utility"],
    "requirements": ["requests", "beautifulsoup4"],
    "environment_vars": ["API_KEY"]
}
```

#### Functions (Tier 2) - Behavior Modification
Functions modify the behavior of the platform, such as filtering content, transforming responses, or customizing UI behavior.

**Use Cases:**
- Content filtering and moderation
- Response transformation
- Custom prompt engineering
- UI behavior modification
- Analytics and logging

**Example Function:**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import re

class BaseFunction(ABC):
    """Base class for function extensions."""
    
    def __init__(self, name: str, description: str, priority: int = 10):
        self.name = name
        self.description = description
        self.priority = priority  # Execution order (lower = earlier)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request (optional)."""
        return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing response (optional)."""
        return response

class ContentFilterFunction(BaseFunction):
    """Function to filter and moderate content."""
    
    def __init__(self):
        super().__init__(
            name="content_filter",
            description="Filter content based on custom rules",
            priority=5  # Run early in the pipeline
        )
        self.filter_rules = [
            {"pattern": r"\b(password|secret|key)\b", "action": "redact"},
            {"pattern": r"\b\d{4}-\d{4}-\d{4}-\d{4}\b", "action": "redact"},  # Credit cards
        ]
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive content from responses."""
        content = response.get("content", "")
        
        for rule in self.filter_rules:
            if rule["action"] == "redact":
                content = re.sub(
                    rule["pattern"], 
                    "[REDACTED]", 
                    content, 
                    flags=re.IGNORECASE
                )
        
        response["content"] = content
        response["filtered"] = True
        
        return response

# Function registration
def register_function():
    return ContentFilterFunction()

FUNCTION_METADATA = {
    "name": "content_filter",
    "version": "1.0.0",
    "author": "Security Team",
    "description": "Filters sensitive content from responses",
    "category": "security",
    "priority": 5
}
```

#### Pipelines (Tier 3) - Complex Workflows
Pipelines orchestrate complex multi-step workflows involving multiple models, tools, and data sources.

**Use Cases:**
- Multi-agent research workflows
- Complex data processing pipelines
- Integration with external systems
- Automated content generation
- Advanced AI workflows

**Example Pipeline:**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import asyncio

class BasePipeline(ABC):
    """Base class for pipeline extensions."""
    
    def __init__(self, name: str, description: str, config: Dict[str, Any]):
        self.name = name
        self.description = description
        self.config = config
        self.steps = self._parse_steps(config.get("steps", []))
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        pass
    
    def _parse_steps(self, steps_config: List[Dict]) -> List[Dict]:
        """Parse and validate step configuration."""
        return steps_config

class ResearchPipeline(BasePipeline):
    """Multi-agent research and synthesis pipeline."""
    
    def __init__(self):
        config = {
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
                    "model": "llama3:8b",
                    "prompt": "Analyze these search results for key insights: ${search_step.output}",
                    "depends_on": ["search_step"]
                },
                {
                    "id": "synthesize_step",
                    "type": "llm_call",
                    "model": "claude-3-sonnet",
                    "prompt": "Synthesize the analysis into a comprehensive report: ${analyze_step.output}",
                    "depends_on": ["analyze_step"]
                }
            ]
        }
        
        super().__init__(
            name="research_pipeline",
            description="Multi-agent research and synthesis workflow",
            config=config
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research pipeline."""
        context = {"input": input_data}
        completed_steps = {}
        
        try:
            for step in self.steps:
                # Check dependencies
                if not self._dependencies_met(step, completed_steps):
                    continue
                
                # Execute step
                result = await self._execute_step(step, context)
                completed_steps[step["id"]] = result
                context[step["id"]] = {"output": result}
            
            return {
                "success": True,
                "result": completed_steps.get("synthesize_step", {}),
                "metadata": {
                    "steps_completed": len(completed_steps),
                    "execution_time": context.get("total_time", 0)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "completed_steps": list(completed_steps.keys())
            }
    
    async def _execute_step(self, step: Dict, context: Dict) -> Any:
        """Execute an individual pipeline step."""
        step_type = step["type"]
        
        if step_type == "tool_call":
            return await self._execute_tool_step(step, context)
        elif step_type == "llm_call":
            return await self._execute_llm_step(step, context)
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    def _dependencies_met(self, step: Dict, completed: Dict) -> bool:
        """Check if step dependencies are satisfied."""
        depends_on = step.get("depends_on", [])
        return all(dep in completed for dep in depends_on)

# Pipeline registration
def register_pipeline():
    return ResearchPipeline()

PIPELINE_METADATA = {
    "name": "research_pipeline",
    "version": "1.0.0",
    "author": "Research Team",
    "description": "Multi-agent research and synthesis workflow",
    "category": "research",
    "complexity": "high",
    "estimated_time": "2-5 minutes",
    "requirements": ["web_search_tool", "llama3", "claude-3"]
}
```

### Extension Development Workflow

#### 1. Planning Phase
- **Define Purpose**: Clear problem statement and solution
- **Choose Type**: Tool, Function, or Pipeline
- **Design Interface**: Input/output schema
- **Plan Testing**: Test cases and validation

#### 2. Development Phase
```bash
# Create extension directory
mkdir -p extensions/my_extension
cd extensions/my_extension

# Create extension files
touch __init__.py
touch main.py
touch requirements.txt
touch README.md
touch tests.py
```

**Extension Structure:**
```
extensions/my_extension/
├── __init__.py
├── main.py              # Main extension code
├── config.py            # Configuration handling
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container definition (if needed)
├── README.md           # Documentation
├── tests.py            # Unit tests
├── examples/           # Usage examples
│   └── example_usage.py
└── docs/               # Additional documentation
    └── api.md
```

#### 3. Testing Phase
```python
# tests.py
import pytest
from main import MyExtension

class TestMyExtension:
    
    @pytest.fixture
    def extension(self):
        return MyExtension()
    
    def test_extension_initialization(self, extension):
        assert extension.name is not None
        assert extension.description is not None
    
    @pytest.mark.asyncio
    async def test_extension_execution(self, extension):
        result = await extension.execute(test_data="sample")
        assert result["success"] is True
        assert "output" in result
    
    def test_extension_schema(self, extension):
        schema = extension.schema
        assert "type" in schema
        assert "function" in schema
```

#### 4. Documentation Phase
```markdown
# My Extension

## Description
Brief description of what this extension does.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from my_extension import MyExtension

extension = MyExtension()
result = await extension.execute(param="value")
```

## Configuration
Environment variables and configuration options.

## API Reference
Detailed API documentation.

## Examples
Practical usage examples.
```

#### 5. Submission Phase
```bash
# Validate extension
python -m pytest tests.py

# Package extension
tar -czf my_extension.tar.gz .

# Submit to marketplace
curl -X POST http://localhost:8001/api/v2/extensions/submit \
  -H "Authorization: Bearer $TOKEN" \
  -F "extension=@my_extension.tar.gz"
```

### Extension Best Practices

#### Security
- **Input Validation**: Validate all inputs thoroughly
- **Sandboxing**: Use containers for isolation
- **Secrets Management**: Use environment variables
- **Error Handling**: Graceful error handling
- **Logging**: Comprehensive logging for debugging

#### Performance
- **Async Operations**: Use async/await for I/O
- **Resource Limits**: Respect memory and CPU limits
- **Caching**: Cache expensive operations
- **Timeouts**: Implement reasonable timeouts
- **Cleanup**: Proper resource cleanup

#### Maintainability
- **Documentation**: Comprehensive documentation
- **Testing**: High test coverage
- **Versioning**: Semantic versioning
- **Backwards Compatibility**: Maintain API compatibility
- **Error Messages**: Clear, actionable error messages

---

## Community Guidelines

### Code of Conduct

#### Our Pledge
We pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

#### Standards
**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Harassment, discrimination, or offensive comments
- Trolling, insulting comments, and personal attacks
- Public or private harassment
- Publishing private information without permission
- Other conduct considered inappropriate in a professional setting

#### Enforcement
Community leaders will fairly and consistently enforce these guidelines. Consequences for violations may include warnings, temporary bans, or permanent exclusion from the community.

### Communication Channels

#### GitHub Issues
- **Bug Reports**: Use issue templates
- **Feature Requests**: Describe use case and benefits
- **Questions**: Check existing issues first
- **Discussions**: Use GitHub Discussions for open-ended topics

#### Discord Community (Future)
- **General Discussion**: Community chat and support
- **Development**: Technical discussions
- **Extensions**: Extension development and sharing
- **Events**: Community events and announcements

#### Email
- **Security Issues**: security@ollama-workbench.org
- **General Inquiries**: hello@ollama-workbench.org
- **Business**: business@ollama-workbench.org

### Recognition

#### Contributors
We recognize and appreciate all forms of contribution:
- **Code Contributors**: Listed in CONTRIBUTORS.md
- **Documentation Contributors**: Credited in documentation
- **Bug Reporters**: Acknowledged in release notes
- **Community Helpers**: Featured in community highlights

#### Maintainers
Active contributors may be invited to become maintainers with additional responsibilities:
- **Code Review**: Review and approve pull requests
- **Issue Triage**: Label and prioritize issues
- **Community Support**: Help answer questions
- **Release Management**: Participate in release planning

---

## Getting Help

### Documentation
- **User Guide**: Comprehensive user documentation
- **API Reference**: Complete API documentation
- **Tutorials**: Step-by-step learning materials
- **FAQ**: Common questions and answers

### Community Support
- **GitHub Discussions**: Community Q&A
- **Discord**: Real-time community chat
- **Stack Overflow**: Tag questions with `ollama-workbench`

### Bug Reports
When reporting bugs, include:
- **Environment**: OS, Python version, dependencies
- **Steps to Reproduce**: Clear reproduction steps
- **Expected vs Actual**: What should happen vs what does
- **Logs**: Relevant error messages and logs
- **Screenshots**: If applicable

### Feature Requests
When requesting features:
- **Use Case**: Why is this feature needed?
- **Benefit**: Who will benefit and how?
- **Implementation**: High-level implementation ideas
- **Alternatives**: Other solutions you've considered

---

Thank you for contributing to Ollama Workbench! Your contributions help make this platform better for everyone. Whether you're fixing bugs, adding features, improving documentation, or helping other users, every contribution is valuable to our community.

For questions about contributing, please reach out through our community channels or create an issue on GitHub. We're here to help you make your first contribution and beyond!