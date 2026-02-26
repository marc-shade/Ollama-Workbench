# Ollama Workbench Observability Integration

## Overview

This implementation adds comprehensive observability capabilities to Ollama Workbench using **Opik**, providing real-time insights into LLM operations, RAG systems, and agentic workflows.

## 🚀 Quick Start

### 1. Install Opik

```bash
# Make the installation script executable
chmod +x install_observability.sh

# Run the installation script
./install_observability.sh
```

Or install manually:
```bash
pip install opik>=0.2.0
```

### 2. Restart Ollama Workbench

```bash
streamlit run main.py
```

### 3. Access the Dashboard

Navigate to: **Maintain** → **Observability Dashboard**

## 📊 Features

### Core Capabilities
- **🔍 LLM Call Tracing**: Automatic capture of all Ollama API interactions
- **📈 Performance Monitoring**: Response time, token usage, and throughput metrics  
- **🚨 Error Tracking**: Comprehensive error monitoring and debugging
- **🔄 RAG Pipeline Observability**: End-to-end tracking of retrieval and generation
- **🤖 Multi-Agent Workflow Monitoring**: Agent decision-making and collaboration tracking

### Dashboard Components
- **Real-time Metrics**: Live system performance and activity
- **Trace Analysis**: Detailed execution traces and span-level insights
- **Performance Trends**: Historical performance analysis and optimization insights
- **Error Monitoring**: Error detection, categorization, and alerting
- **System Health**: Overall system status and configuration overview

## ⚙️ Configuration

### Local Mode (Default)
No API key required - all data stays local:
```python
# Automatically configured for local usage
project_name = "ollama-workbench"
local_mode = True
```

### Cloud Mode (Optional)
For team collaboration and advanced analytics:

1. **Get an API key** from [Comet ML Opik](https://www.comet.ml/opik)
2. **Configure in dashboard** or set environment variable:
   ```bash
   export OPIK_API_KEY='your-api-key-here'
   ```
3. **Restart the application**

### Privacy Settings
Configure data collection levels:
- **Hash prompts**: Enable for sensitive deployments
- **Truncate responses**: Limit response length capture
- **Local processing**: Keep all data on-premises

## 🔧 Technical Implementation

### Enhanced Functions

#### LLM Calls with Observability
```python
# Automatic tracing is now built into call_ollama_endpoint
response, context, eval_count, eval_duration, metrics = call_ollama_endpoint(
    model="llama3",
    prompt="Your prompt here",
    temperature=0.7
)
# Traces are automatically captured with metadata
```

#### RAG Operations (Coming in Phase 2)
```python
from observability import trace_rag_operation

@trace_rag_operation()
def your_rag_function(query, collection):
    # Your RAG implementation
    pass
```

#### Agent Workflows (Coming in Phase 2)
```python
from observability import trace_agent_operation

@trace_agent_operation()
def your_agent_function():
    # Your agent implementation  
    pass
```

### Data Collection

**Automatically Captured:**
- Model name and provider
- Request parameters (temperature, max_tokens, etc.)
- Response times and token counts
- Error information and context
- System resource usage
- User interaction patterns

**Configurable Privacy:**
- Prompt hashing for sensitive data
- Response truncation options
- Metadata-only collection modes
- Local vs cloud storage options

## 📈 Business Value

### For 2 Acre Studios
- **Operational Excellence**: Monitor client deployments for performance issues
- **ROI Demonstration**: Detailed usage analytics and performance reports
- **Product Development**: Data-driven feature prioritization and optimization
- **Quality Assurance**: Comprehensive testing through automated monitoring

### For Small Business Clients
- **Cost Optimization**: Identify most cost-effective models for specific use cases
- **Performance Optimization**: Automatically detect and resolve performance issues
- **Resource Planning**: Predict scaling requirements and optimize allocation
- **Transparency**: Detailed reporting for stakeholders and compliance

### For Nonprofit Organizations
- **Resource Management**: Track and optimize AI tool usage across programs
- **Impact Demonstration**: Generate reports for grant applications and funding
- **Compliance**: Maintain audit trails for regulated activities
- **Governance**: Monitor data usage and access patterns

## 🎯 Success Metrics

### Technical Performance
- **99%+ trace coverage** of LLM interactions
- **<5% performance impact** from observability overhead
- **<1% false positive rate** for performance alerts
- **<2 seconds dashboard load time** for real-time metrics

### Business Impact
- **20%+ improvement** in resource utilization
- **15%+ reduction** in AI operation costs
- **75%+ faster** issue resolution
- **90%+ accuracy** in capacity planning

## 🛠️ Implementation Status

### ✅ Phase 1: Complete (Current)
- [x] Core Opik integration
- [x] Basic LLM call tracing
- [x] Performance metrics enhancement
- [x] Configuration system
- [x] Enhanced dashboard
- [x] Error tracking and debugging

### 🚧 Phase 2: Advanced Workflows (Next)
- [ ] RAG pipeline instrumentation
- [ ] Multi-agent workflow monitoring
- [ ] Real-time alerting system
- [ ] Advanced analytics

### 📋 Phase 3: Intelligence & Optimization (Future)
- [ ] Predictive analytics
- [ ] A/B testing framework
- [ ] Intelligent model routing
- [ ] Cost optimization recommendations

## 🔍 Troubleshooting

### Common Issues

**1. Opik not available error**
```bash
# Install Opik
pip install opik>=0.2.0
# Restart application
```

**2. Configuration not loading**
```bash
# Check data directory exists
mkdir -p data
# Recreate config by running dashboard once
```

**3. Traces not appearing**
```bash
# Check if observability is enabled
# Navigate to dashboard → Configuration → Enable Opik Integration
```

**4. Performance impact**
```bash
# Reduce trace frequency in config
# Disable detailed metrics if not needed
# Use local mode for better performance
```

### Debug Mode
Enable detailed logging:
```bash
export OPIK_DEBUG=true
export OBSERVABILITY_LOG_LEVEL=DEBUG
```

## 📚 Additional Resources

- **Opik Documentation**: https://www.comet.ml/docs/opik
- **Ollama Workbench PRD**: See PRD..md for detailed requirements
- **2 Acre Studios**: https://2acrestudios.com

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the configuration in the dashboard
3. Check application logs for error messages
4. Contact 2 Acre Studios for enterprise support

## 🔄 Updates

This observability integration is actively maintained and enhanced. Check for updates in:
- New releases of Ollama Workbench
- Opik package updates
- Configuration setting improvements
- Dashboard feature additions

---

**Built by 2 Acre Studios for the Ollama Workbench community** 🦙✨
