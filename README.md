[![Version](https://img.shields.io/github/v/release/marc-shade/Ollama-Workbench?style=flat-square)](https://github.com/marc-shade/Ollama-Workbench/releases)
[![Stars](https://img.shields.io/github/stars/marc-shade/Ollama-Workbench?style=flat-square)](https://github.com/marc-shade/Ollama-Workbench/stargazers)
[![Forks](https://img.shields.io/github/forks/marc-shade/Ollama-Workbench?style=flat-square)](https://github.com/marc-shade/Ollama-Workbench/network/members)
[![License](https://img.shields.io/github/license/marc-shade/Ollama-Workbench?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.ai)

# 🦙 Ollama Workbench - Local AI Workbench

> **Looking for the next generation?** Check out [Ollama Workbench 2.0](https://github.com/marc-shade/Ollama-Workbench-2) — a SvelteKit + Tauri rewrite with a native desktop app, MCP Studio, and visual workflow builder.

<img src="assets/ollama-workbench.jpg" width="300" align="right" alt="Ollama Workbench - No prob-llama" />

**Ollama Workbench** is a comprehensive platform for managing, testing, and using AI models from the Ollama library and external providers (OpenAI, Groq, Mistral). It bundles chat (text, voice, vision), retrieval-augmented generation, multi-agent workflows, model management and benchmarking, and an OpenAI-compatible API - all running locally against your own Ollama server.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security](https://img.shields.io/badge/Security-Hardening%20Toolkit-green.svg)](#security-features)
[![Observability](https://img.shields.io/badge/Observability-Opik%20Integration-orange.svg)](#observability-features)

## 🚀 Quick Start

**One-Command Setup:**
```bash
git clone https://github.com/marc-shade/Ollama-Workbench.git
cd Ollama-Workbench
python3 scripts/setup_workbench.py
```

**Start the Platform:**
```bash
./start_workbench.sh  # Unix/Linux/macOS
```

**Access the Interface:**
- Web UI: http://localhost:8501
- OpenAI-compatible API: http://localhost:8000/v1 (starts with the app; toggle with `ENABLE_OPENAI_COMPAT`)
- TTS server (optional, standalone): http://localhost:5002 (`tts_server/start_tts_server.sh`)
- Default login: No authentication required initially (configurable)

## ✨ What's New (July 2026)

- **Modern dark UI** built on Streamlit's native theming (`.streamlit/config.toml`) — orange accent, themed sidebar, rounded widgets; customize the whole look by editing one file
- **OpenAI-compatible API server** starts with the app on port 8000, so any OpenAI client can talk to your local Ollama models
- **Hardened live paths**: chat streaming, voice chat, RAG fallback, workflows, and telemetry all repaired and verified against a running Ollama server
- **1,076-test suite, fully green** — hermetic (no live server required), run it with `venv/bin/python -m pytest tests/`

---

## 📋 Table of Contents

- [✨ What's New](#-whats-new-july-2026)
- [🌟 Key Features](#-key-features)
- [🛡️ Security Features](#️-security-features)
- [📊 Observability Features](#-observability-features)
- [💬 Chat & AI Interaction](#-chat--ai-interaction)
- [⚙️ Advanced Workflows](#️-advanced-workflows)
- [🗄️ Knowledge Management](#️-knowledge-management)
- [🛠️ Model Management](#️-model-management)
- [📊 Testing & Evaluation](#-testing--evaluation)
- [🔧 Installation](#-installation)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)

---

## 🌟 Key Features

### 🏢 **Solid Foundations**
- **Encrypted API-key storage** (AES-256-GCM) and integrity-hashed audit logging
- **Security toolkit included**: JWT auth, RBAC, and encryption modules ready to wire up (see [Security Features](#️-security-features) for what's active by default)
- **Observability** with optional Opik integration and local performance monitoring
- **Streamlit UI + Flask API services** with an OpenAI-compatible endpoint

### 🤖 **Multi-Provider AI Integration**
- **Ollama Models** - Local LLM execution with full privacy
- **OpenAI API** - GPT-3.5, GPT-4, and latest models
- **Groq** - Ultra-fast inference with Llama and Mixtral
- **Mistral AI** - European AI excellence
- **Unified Interface** - Single platform for all providers

### 🔄 **Advanced Workflow Orchestration**
- **Multi-Agent Systems** with specialized AI agents
- **Visual Workflow Builder** (Nodes) for complex pipelines
- **Project Management** with AI-assisted planning
- **Brainstorming Sessions** with collaborative AI teams
- **Research Automation** with multi-source intelligence

---

## 🛡️ Security Features

An honest map of what ships in the `security/` package: what runs by default,
what is included but must be wired up, and what is not implemented.

### ✅ **Active by default**
- **Encrypted API-key storage** - provider keys are encrypted at rest with
  AES-256-GCM (`cryptography` library) when `ENABLE_ENCRYPTION` is on
- **Security audit logging** - configuration changes and security events are
  written as structured audit records, each with a SHA-256 integrity hash
  (`ENABLE_AUDIT_LOGGING`)
- **Security framework self-check** - configuration validation plus a
  self-assessment report that scores the current settings against GDPR /
  HIPAA / SOX / ISO27001 checklists (a gap report, not a compliance
  certification)

### 🧰 **Included, but not wired into the UI yet**
The app currently runs **without authentication** - these modules exist in
`security/` and are importable, but no login gate protects the Streamlit UI:
- **JWT authentication** (`security/authentication.py`) - bcrypt password
  hashing, JWT sessions, and account lockout after configurable failed
  attempts. The Streamlit auth UI hook exists but is disabled by default.
- **Role-Based Access Control** (`security/access_control.py`) - role and
  permission model (guest/user/admin) ready to enforce, currently not
  called by any UI path
- **Key rotation** (`security/encryption.py`, `security/security_config.py`) -
  `rotate_keys()` / `rotate_secrets()` methods with a rotation-interval
  setting; rotation is manual (nothing schedules it automatically)

### ❌ **Not implemented (do not rely on these)**
- Multi-factor authentication - the user model has MFA fields, but there is
  no TOTP generation or verification code
- Transport encryption - the app serves plain HTTP on localhost; put a TLS
  proxy in front for remote access
- Tamper-proof logs - audit events carry integrity hashes but are not
  chained or signed
- Virus scanning of uploads, intrusion detection, data-loss prevention

**Deployment guidance**: bind to localhost (the default), treat the app as a
single-user local tool, and wire up the auth module before exposing it to a
network you do not control.

---

## 📊 Observability Features

### 📈 **Performance Monitoring**
- **Real-time Metrics** with custom dashboards
- **Performance Alerts** with configurable thresholds
- **Token Usage Tracking** across all providers
- **Response Time Analysis** with percentile breakdowns
- **Resource Utilization** monitoring (CPU, memory, GPU)

### 🔍 **LLM Operation Tracing**
- **Opik Integration** for comprehensive LLM observability
- **Request/Response Tracing** with full context capture
- **Error Tracking** with detailed error analysis
- **Cost Analytics** across different AI providers
- **Usage Pattern Analysis** for optimization insights

### 📊 **Analytics & Reporting**
- **Usage Reports** with trend analysis
- **Model Performance Comparisons** across providers
- **Security Event Analysis** with risk assessment
- **Capacity Planning** with growth projections
- **Export Capabilities** for external analysis

---

## 💬 Chat & AI Interaction

### 🧑‍🔧 **Advanced Agent Configuration**
- **Pre-built Agent Types**: Coder, Analyst, Creative Writer, Researcher
- **Custom Agent Creation** with specialized prompts and behaviors
- **Metacognitive Enhancements**: Chain of Thought, Visualization of Thought
- **Voice & Personality Customization** for tailored interactions
- **Memory Systems** with episodic and semantic memory support

### 🗣️ **Multimodal Capabilities**
- **Text-to-Speech (TTS)** with multiple voice options
- **Speech Recognition** for voice input
- **Vision Model Support** for image analysis and description
- **Document Processing** with OCR and extraction
- **File Upload & Analysis** for various formats

### 📚 **Retrieval Augmented Generation (RAG)**
- **Dynamic Corpus Integration** for contextual enhancement
- **Vector Database Support** with ChromaDB integration
- **Semantic Search** with advanced embedding models
- **Document Chunking** with intelligent splitting strategies
- **Real-time Context Injection** for improved responses

---

## ⚙️ Advanced Workflows

### 🧠 **Multi-Agent Orchestration**
- **Brainstorm Mode**: Collaborative AI teams for ideation
- **Research Workflows**: Multi-source intelligence gathering
- **Build System**: Autonomous software development with specialized agents
- **Project Management**: AI-assisted task planning and execution
- **Quality Assurance**: Automated testing and review processes

### 📊 **Visual Workflow Builder (Nodes)**
- **Drag-and-Drop Interface** for workflow design
- **Pre-built Components**: AI models, data processors, connectors
- **Custom Node Creation** with Python scripting support
- **Conditional Logic** with branching and loops
- **Real-time Execution** with progress monitoring

### 🔗 **Integration Capabilities**
- **API Gateway** with rate limiting and authentication
- **Webhook Support** for external integrations
- **Database Connectors** for various data sources
- **File System Integration** with secure access controls
- **Third-party Tool Integration** via standardized protocols

---

## 🗄️ Knowledge Management

### 📂 **Corpus Management**
- **Text Corpus Creation** from various sources
- **Web Content Integration** with intelligent extraction
- **Document Processing** with format support (PDF, DOCX, TXT, MD)
- **Knowledge Graph Construction** for relationship mapping
- **Version Control** for corpus evolution tracking

### 🕸️ **Data Sources**
- **Web Scraping** with respect for robots.txt
- **API Data Integration** from external services
- **Database Connectivity** for structured data
- **File Upload System** for corpus ingestion
- **Real-time Data Feeds** for dynamic knowledge

### 🔍 **Search & Discovery**
- **Full-text Search** across all knowledge bases
- **Semantic Search** with embedding-based similarity
- **Faceted Search** with filters and categories
- **Search Analytics** for usage optimization
- **Personalized Recommendations** based on user patterns

---

## 🛠️ Model Management

### 📋 **Model Operations**
- **Local Model Management**: List, download, update, remove Ollama models
- **Model Information**: Detailed specs, capabilities, and performance metrics
- **Health Monitoring**: Model availability and performance tracking
- **Usage Analytics**: Token consumption and cost analysis
- **Model Comparison**: Side-by-side performance evaluation

### ⚙️ **Server Configuration**
- **Ollama Server Management**: Start, stop, configure, monitor
- **Resource Allocation**: CPU, memory, and GPU settings
- **Concurrency Controls**: Request queuing and parallel processing
- **Performance Tuning**: Optimization for specific workloads
- **Backup & Recovery**: Model and configuration backup

### ☁️ **External Provider Management**
- **API Key Management**: Secure storage and rotation
- **Rate Limit Monitoring**: Usage tracking and alerts
- **Cost Management**: Budget controls and spending alerts
- **Provider Comparison**: Performance and cost analysis
- **Failover Configuration**: Automatic provider switching

---

## 📊 Testing & Evaluation

### 🧪 **Model Testing**
- **Feature Testing**: JSON handling, function calling, structured output
- **Performance Benchmarking**: Speed, accuracy, and consistency
- **Comparative Analysis**: Multi-model evaluation on same tasks
- **Regression Testing**: Automated testing for model updates
- **Custom Test Suites**: Domain-specific evaluation frameworks

### 🎯 **Quality Assurance**
- **Response Quality Metrics**: Coherence, relevance, factuality
- **Contextual Understanding**: Multi-turn conversation evaluation
- **Bias Detection**: Automated bias and fairness testing
- **Safety Evaluation**: Content safety and harmfulness detection
- **Performance Profiling**: Resource usage and optimization

### 👁️ **Vision Model Testing**
- **Image Understanding**: Object detection and scene analysis
- **OCR Capabilities**: Text extraction from images
- **Visual Question Answering**: Image-based reasoning
- **Art and Design**: Creative image analysis
- **Medical Imaging**: Specialized vision model testing

---

## 🔧 Installation

### 📋 **System Requirements**
- **Python**: 3.10+
- **Operating System**: Windows, macOS, Linux
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 10GB+ free space
- **Network**: Internet connection for model downloads

### ⚡ **Automated Setup** (Recommended)

```bash
# Clone the repository
git clone https://github.com/marc-shade/Ollama-Workbench.git
cd Ollama-Workbench

# Run automated setup
python3 scripts/setup_workbench.py
```

This script automatically:
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Downloads and configures Ollama
- ✅ Initializes security framework
- ✅ Sets up directories and configuration
- ✅ Creates startup scripts
- ✅ Runs basic tests

### 🐍 **Manual Setup**

```bash
# Create virtual environment (Python 3.11 recommended; 3.10+ supported)
python3 -m venv venv
source venv/bin/activate  # Unix/Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh  # Unix/Linux/macOS
# For Windows: Download from https://ollama.ai/download

# Start Ollama server
ollama serve

# Download a default model
ollama pull llama3.2:1b

# Start the workbench
streamlit run main.py
```

### 🔧 **Configuration**

Initial configuration is created automatically, but you can customize:

```json
{
  "OLLAMA_HOST": "http://localhost:11434",
  "WORKBENCH_PORT": 8501,
  "ENABLE_OPENAI_COMPAT": true,
  "ENABLE_ENHANCED_SECURITY": true,
  "ENABLE_AUTH": false,
  "ENABLE_RBAC": true,
  "ENABLE_AUDIT_LOGGING": true,
  "ENABLE_ENCRYPTION": true,
  "ENABLE_OBSERVABILITY": true
}
```

The UI theme lives in `.streamlit/config.toml` — colors, fonts, radii, and the
sidebar palette are all editable there (Streamlit >= 1.47 native theming).

---

## 📚 Documentation

### 📖 **Complete Documentation Suite**

- **[Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)** - System design and architecture
- **[Security & Compliance](docs/SECURITY_COMPLIANCE.md)** - Security features and compliance
- **[API Documentation](docs/API_DOCUMENTATION.md)** - Pipeline framework and APIs
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment strategies
- **[Contributing Guide](CONTRIBUTING.md)** - Development guidelines
- **[User Stories](docs/USER_STORIES.md)** - Detailed use cases and workflows
- **[UX Design Guide](docs/UX_DESIGN_GUIDE.md)** - Interface design principles
- **[Observability Specification](docs/OBSERVABILITY_SPECIFICATION.md)** - Monitoring and tracing
- **[Glossary](docs/GLOSSARY.md)** - Domain terminology and concepts
- **[Apple Silicon Setup](docs/APPLE_SILICON_SETUP.md)** - macOS arm64 notes

---

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### 🛠️ **Development Setup**

```bash
# Clone and setup development environment
git clone https://github.com/marc-shade/Ollama-Workbench.git
cd Ollama-Workbench
python3 scripts/setup_workbench.py

# Install development dependencies
pip install -r requirements-dev.txt

# Run the test suite (1,076 tests, hermetic - no live Ollama needed)
python -m pytest tests/

# Run linting
ruff check .
flake8 .
```

### 📝 **Contributing Areas**

- 🐛 **Bug Reports**: Help us identify and fix issues
- 🚀 **Feature Requests**: Suggest new capabilities
- 📖 **Documentation**: Improve guides and tutorials
- 🧪 **Testing**: Add test cases and improve coverage
- 🔒 **Security**: Enhance security features and audit code
- 🎨 **UI/UX**: Improve user interface and experience
- 🔌 **Integrations**: Add support for new AI providers

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support & Community

- **📧 Issues**: [GitHub Issues](https://github.com/marc-shade/Ollama-Workbench/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/marc-shade/Ollama-Workbench/discussions)
- **📚 Documentation**: Complete guides in `/docs` directory
- **🐛 Bug Reports**: Use issue templates for better tracking
- **💡 Feature Requests**: Share your ideas for improvements

---

## 🙏 Acknowledgments

- **Ollama Team** - For the excellent local LLM runtime
- **Streamlit** - For the amazing web app framework
- **OpenAI, Groq, Mistral** - For their powerful AI APIs
- **Opik** - For comprehensive LLM observability
- **Contributors** - For making this project better every day

---

**Made with ❤️ by the Ollama Workbench Community**

*A local-first workbench for serious work with open models.*