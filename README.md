[![Version](https://img.shields.io/github/v/release/marc-shade/Ollama-Workbench?style=flat-square)](https://github.com/marc-shade/Ollama-Workbench/releases)
[![Stars](https://img.shields.io/github/stars/marc-shade/Ollama-Workbench?style=flat-square)](https://github.com/marc-shade/Ollama-Workbench/stargazers)
[![Forks](https://img.shields.io/github/forks/marc-shade/Ollama-Workbench?style=flat-square)](https://github.com/marc-shade/Ollama-Workbench/network/members)
[![License](https://img.shields.io/github/license/marc-shade/Ollama-Workbench?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.ai)

# 🦙 Ollama Workbench - Enterprise AI Platform

> ## :rocket: **[Ollama Workbench 2.0 is here!](https://github.com/marc-shade/Ollama-Workbench-2)**
>
> A complete rewrite with **SvelteKit + Tauri** featuring:
> - Modern responsive UI with dark/light themes
> - Native desktop app (macOS, Windows, Linux)
> - MCP Studio for building and testing MCP servers
> - Visual multi-agent workflow builder
> - Tools debugger with model comparison
> - Prompt Lab with A/B testing and version control
>
> **[Try Ollama Workbench 2.0 →](https://github.com/marc-shade/Ollama-Workbench-2)**

<img src="https://2acrestudios.com/wp-content/uploads/2024/06/00001-2881912941.png" style="width: 300px;" align="right" />

**Ollama Workbench** is a comprehensive, enterprise-grade platform for managing, testing, and utilizing AI models from the Ollama library and external providers. Built with security, scalability, and observability at its core, it provides advanced features for AI agent orchestration, workflow automation, and collaborative AI development.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security](https://img.shields.io/badge/Security-Enterprise%20Grade-green.svg)](#security-features)
[![Observability](https://img.shields.io/badge/Observability-Opik%20Integration-orange.svg)](#observability-features)

## 🚀 Quick Start

**One-Command Setup:**
```bash
git clone https://github.com/marc-shade/Ollama-Workbench.git
cd Ollama-Workbench
python setup_workbench.py
```

**Start the Platform:**
```bash
./start_workbench.sh  # Unix/Linux/macOS
start_workbench.bat   # Windows
```

**Access the Interface:**
- Web UI: http://localhost:8501
- Default login: No authentication required initially (configurable)

---

## 📋 Table of Contents

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

### 🏢 **Enterprise-Ready Architecture**
- **Zero-Trust Security** with comprehensive RBAC system
- **End-to-End Encryption** for data at rest and in transit
- **Comprehensive Audit Logging** for compliance (GDPR, HIPAA, SOX, ISO27001)
- **Advanced Observability** with Opik integration and performance monitoring
- **Scalable Microservices** architecture with FastAPI and Streamlit

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

### 🔐 **Authentication & Authorization**
- **JWT-based Authentication** with configurable session management
- **Multi-Factor Authentication** support (foundation ready)
- **Role-Based Access Control (RBAC)** with granular permissions
- **Account Lockout Protection** with rate limiting
- **Password Policy Enforcement** with complexity requirements

### 🔒 **Data Protection**
- **AES-256-GCM Encryption** for sensitive data
- **RSA Encryption** for key exchange and secrets
- **Automatic Key Rotation** with configurable intervals
- **Secure API Key Storage** with encryption at rest
- **Data Classification** and loss prevention

### 📋 **Compliance & Auditing**
- **Comprehensive Audit Trails** with tamper-proof logging
- **Compliance Templates** for GDPR, HIPAA, SOX, ISO27001
- **Real-time Security Monitoring** with threat detection
- **Automated Compliance Reporting** with violation alerts
- **Data Retention Policies** with automated cleanup

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
- **File Upload System** with virus scanning
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
- **Python**: 3.8+ (recommended: 3.11)
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
python setup_workbench.py
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
# Create virtual environment
python -m venv venv
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
  "ENABLE_ENHANCED_SECURITY": true,
  "ENABLE_AUTH": false,
  "ENABLE_RBAC": true,
  "ENABLE_AUDIT_LOGGING": true,
  "ENABLE_ENCRYPTION": true,
  "ENABLE_OBSERVABILITY": true
}
```

---

## 📚 Documentation

### 📖 **Complete Documentation Suite**

- **[Technical Architecture](TECHNICAL_ARCHITECTURE.md)** - System design and architecture
- **[Security & Compliance](SECURITY_COMPLIANCE.md)** - Security features and compliance
- **[API Documentation](API_DOCUMENTATION.md)** - Pipeline framework and APIs
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment strategies
- **[Contributing Guide](CONTRIBUTING.md)** - Development guidelines
- **[User Stories](USER_STORIES.md)** - Detailed use cases and workflows
- **[Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)** - Development timeline
- **[UX Design Guide](UX_DESIGN_GUIDE.md)** - Interface design principles
- **[Observability Specification](OBSERVABILITY_SPECIFICATION.md)** - Monitoring and tracing
- **[Glossary](GLOSSARY.md)** - Domain terminology and concepts

### 🎓 **Getting Started Guides**

1. **[Quick Start Guide](#-quick-start)** - Get up and running in minutes
2. **[First Steps Tutorial](docs/tutorials/first-steps.md)** - Basic platform usage
3. **[Security Setup Guide](docs/tutorials/security-setup.md)** - Enable authentication and RBAC
4. **[Workflow Creation Tutorial](docs/tutorials/workflow-creation.md)** - Build your first workflow
5. **[API Integration Guide](docs/tutorials/api-integration.md)** - Connect external services

### 🔧 **Advanced Topics**

- **[Custom Agent Development](docs/advanced/custom-agents.md)** - Build specialized AI agents
- **[Security Hardening](docs/advanced/security-hardening.md)** - Production security setup
- **[Performance Optimization](docs/advanced/performance-optimization.md)** - Scale and optimize
- **[Compliance Configuration](docs/advanced/compliance-setup.md)** - Meet regulatory requirements
- **[Monitoring & Alerting](docs/advanced/monitoring-setup.md)** - Comprehensive observability

---

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### 🛠️ **Development Setup**

```bash
# Clone and setup development environment
git clone https://github.com/marc-shade/Ollama-Workbench.git
cd Ollama-Workbench
python setup_workbench.py

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
ruff check .
flake8 .

# Run security checks
python -m bandit -r .
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

*Transform your AI development with enterprise-grade tools, security, and observability.*