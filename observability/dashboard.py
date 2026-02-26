# observability/dashboard.py

"""
Enhanced observability dashboard for Ollama Workbench.
Integrates existing performance metrics with Opik observability data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import logging

from .config import observability_config, is_local_mode, get_opik_project_name
from .opik_integration import is_opik_enabled, OPIK_AVAILABLE

# Import existing performance metrics
try:
    from performance_metrics import load_metrics_data, record_metrics
    LEGACY_METRICS_AVAILABLE = True
except ImportError:
    LEGACY_METRICS_AVAILABLE = False
    def load_metrics_data():
        return []
    def record_metrics(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)


def enhanced_observability_dashboard():
    """Enhanced observability dashboard with Opik integration"""
    st.title("🔍 Observability Dashboard")
    
    # Status overview
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            opik_status = "✅ Active" if is_opik_enabled() else "❌ Inactive"
            st.metric("Opik Integration", opik_status)
        
        with col2:
            mode = "Local" if is_local_mode() else "Cloud"
            st.metric("Mode", mode)
        
        with col3:
            project = get_opik_project_name()
            st.metric("Project", project)
        
        with col4:
            legacy_status = "✅ Available" if LEGACY_METRICS_AVAILABLE else "❌ Unavailable"
            st.metric("Legacy Metrics", legacy_status)
    
    # Configuration section
    with st.expander("⚙️ Configuration", expanded=False):
        show_configuration_panel()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-time Metrics", 
        "🔍 Trace Analysis", 
        "📈 Performance Trends",
        "🚨 Error Monitoring",
        "📋 System Health"
    ])
    
    with tab1:
        show_realtime_metrics()
    
    with tab2:
        show_trace_analysis()
    
    with tab3:
        show_performance_trends()
    
    with tab4:
        show_error_monitoring()
    
    with tab5:
        show_system_health()


def show_configuration_panel():
    """Show observability configuration panel"""
    st.subheader("Observability Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Opik Settings**")
        
        # Opik enable/disable
        current_enabled = observability_config.is_opik_enabled()
        enabled = st.checkbox("Enable Opik Integration", value=current_enabled)
        
        if enabled != current_enabled:
            observability_config.set('opik.enabled', enabled)
            observability_config.save_config()
            st.rerun()
        
        # Project name
        project_name = st.text_input(
            "Project Name", 
            value=get_opik_project_name(),
            help="Name of the Opik project for organizing traces"
        )
        
        # API key
        api_key = st.text_input(
            "API Key (optional)", 
            type="password",
            help="Leave empty for local mode"
        )
        
        if st.button("Update Opik Settings"):
            from .config import update_opik_settings
            update_opik_settings(project_name, api_key if api_key else None)
            st.success("Settings updated!")
            st.rerun()
    
    with col2:
        st.markdown("**Privacy Settings**")
        
        # Privacy options
        hash_prompts = st.checkbox(
            "Hash prompts for privacy",
            value=observability_config.should_hash_prompts(),
            help="Hash prompts before sending to observability platform"
        )
        
        truncate_responses = st.checkbox(
            "Truncate long responses",
            value=observability_config.get('privacy.truncate_responses', False),
            help="Truncate responses longer than specified length"
        )
        
        max_length = st.number_input(
            "Max response length",
            min_value=100,
            max_value=10000,
            value=observability_config.get('privacy.max_response_length', 1000),
            help="Maximum length for captured responses"
        )
        
        if st.button("Update Privacy Settings"):
            from .config import configure_privacy_settings
            configure_privacy_settings(hash_prompts, truncate_responses, max_length)
            st.success("Privacy settings updated!")
            st.rerun()


def show_realtime_metrics():
    """Show real-time metrics and current system status"""
    st.subheader("📊 Real-time Metrics")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Load recent metrics data
    metrics_data = load_metrics_data()
    recent_data = [m for m in metrics_data if 'timestamp' in m]
    
    # Sort by timestamp and get recent data
    if recent_data:
        recent_data.sort(key=lambda x: x['timestamp'], reverse=True)
        last_hour_data = []
        cutoff = datetime.now() - timedelta(hours=1)
        
        for item in recent_data:
            try:
                item_time = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                if item_time >= cutoff:
                    last_hour_data.append(item)
            except:
                continue
    else:
        last_hour_data = []
    
    with col1:
        total_calls = len(last_hour_data)
        st.metric("Calls (Last Hour)", total_calls)
    
    with col2:
        if last_hour_data:
            avg_response_time = sum(item.get('response_time', 0) for item in last_hour_data) / len(last_hour_data)
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        else:
            st.metric("Avg Response Time", "N/A")
    
    with col3:
        if last_hour_data:
            total_tokens = sum(item.get('output_tokens', 0) for item in last_hour_data)
            st.metric("Total Tokens (1h)", total_tokens)
        else:
            st.metric("Total Tokens (1h)", "N/A")
    
    with col4:
        error_count = sum(1 for item in last_hour_data if item.get('error_type'))
        st.metric("Errors (1h)", error_count, delta_color="inverse")
    
    # Real-time activity chart
    if last_hour_data:
        st.subheader("Activity Timeline (Last Hour)")
        
        # Create timeline data
        timeline_data = []
        for item in last_hour_data:
            timeline_data.append({
                'timestamp': item['timestamp'],
                'model': item.get('model', 'unknown'),
                'response_time': item.get('response_time', 0),
                'operation_type': item.get('operation_type', 'unknown'),
                'tokens': item.get('output_tokens', 0)
            })
        
        df = pd.DataFrame(timeline_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Timeline chart
        fig = px.scatter(df, x='timestamp', y='response_time', 
                        color='model', size='tokens',
                        hover_data=['operation_type'],
                        title="Recent LLM Calls")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent activity data available.")


def show_trace_analysis():
    """Show detailed trace analysis from Opik"""
    st.subheader("🔍 Trace Analysis")
    
    if not is_opik_enabled():
        st.warning("Opik integration is not enabled. Enable it in the configuration section to see detailed traces.")
        return
    
    if not OPIK_AVAILABLE:
        st.error("Opik package is not installed. Install it with: pip install opik")
        return
    
    # Note: In a real implementation, this would fetch traces from Opik API
    st.info("🚧 Trace analysis will show detailed execution traces from Opik once traces are captured.")
    
    # Placeholder for trace analysis
    st.markdown("""
    **Trace Analysis Features (Coming Soon):**
    - Individual trace inspection
    - Span-level performance analysis
    - Error trace debugging
    - Distributed tracing for multi-agent workflows
    - Custom trace filtering and search
    """)
    
    # Show trace statistics if available
    metrics_data = load_metrics_data()
    if metrics_data:
        st.subheader("Trace Statistics")
        
        # Model usage breakdown
        model_counts = {}
        for item in metrics_data:
            model = item.get('model', 'unknown')
            model_counts[model] = model_counts.get(model, 0) + 1
        
        if model_counts:
            fig = px.pie(values=list(model_counts.values()), 
                        names=list(model_counts.keys()),
                        title="Model Usage Distribution")
            st.plotly_chart(fig, use_container_width=True)


def show_performance_trends():
    """Show performance trends over time"""
    st.subheader("📈 Performance Trends")
    
    # Load and process metrics data
    metrics_data = load_metrics_data()
    
    if not metrics_data:
        st.info("No performance data available yet. Start using the chat to generate metrics.")
        return
    
    # Process data into DataFrame
    processed_data = []
    for item in metrics_data:
        if 'response_time' in item:
            processed_data.append({
                'timestamp': item.get('timestamp', datetime.now().isoformat()),
                'model': item.get('model', 'unknown'),
                'response_time': item.get('response_time', 0),
                'input_tokens': item.get('input_tokens', 0),
                'output_tokens': item.get('output_tokens', 0),
                'operation_type': item.get('operation_type', 'unknown')
            })
    
    if not processed_data:
        st.info("No performance data with response times available.")
        return
    
    df = pd.DataFrame(processed_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    
    # Performance trends
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time trends
        fig = px.line(df, x='timestamp', y='response_time', color='model',
                     title="Response Time Trends")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Token usage trends
        fig = px.line(df, x='timestamp', y='output_tokens', color='model',
                     title="Token Usage Trends")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary table
    st.subheader("Performance Summary by Model")
    summary = df.groupby('model').agg({
        'response_time': ['mean', 'min', 'max', 'std'],
        'output_tokens': ['mean', 'sum'],
        'timestamp': 'count'
    }).round(3)
    
    summary.columns = ['Avg Response Time', 'Min Response Time', 'Max Response Time', 'Std Dev', 
                      'Avg Tokens', 'Total Tokens', 'Total Calls']
    st.dataframe(summary)


def show_error_monitoring():
    """Show error monitoring and alerts"""
    st.subheader("🚨 Error Monitoring")
    
    # Load metrics to find errors
    metrics_data = load_metrics_data()
    error_data = [item for item in metrics_data if item.get('error_type')]
    
    if not error_data:
        st.success("✅ No errors detected in recent activity!")
        return
    
    # Error summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_errors = len(error_data)
        st.metric("Total Errors", total_errors)
    
    with col2:
        # Last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        recent_errors = []
        for item in error_data:
            try:
                item_time = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                if item_time >= cutoff:
                    recent_errors.append(item)
            except:
                continue
        st.metric("Errors (24h)", len(recent_errors))
    
    with col3:
        # Error rate
        total_calls = len(metrics_data)
        error_rate = (total_errors / total_calls * 100) if total_calls > 0 else 0
        st.metric("Error Rate", f"{error_rate:.2f}%")
    
    # Error breakdown
    if error_data:
        st.subheader("Error Breakdown")
        
        error_types = {}
        for item in error_data:
            error_type = item.get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        fig = px.bar(x=list(error_types.keys()), y=list(error_types.values()),
                    title="Error Types")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent errors table
        st.subheader("Recent Errors")
        error_df = pd.DataFrame(error_data)
        error_df = error_df.sort_values('timestamp', ascending=False).head(10)
        st.dataframe(error_df[['timestamp', 'error_type', 'model', 'error_message']])


def show_system_health():
    """Show overall system health and status"""
    st.subheader("📋 System Health")
    
    # Import system monitoring functions
    try:
        from ollama_utils import get_ollama_resource_usage
        resource_usage = get_ollama_resource_usage()
    except:
        resource_usage = {}
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Ollama Server**")
        server_status = resource_usage.get('status', 'Unknown')
        if server_status == 'Running':
            st.success(f"✅ {server_status}")
        else:
            st.error(f"❌ {server_status}")
    
    with col2:
        st.markdown("**Observability**")
        if is_opik_enabled():
            st.success("✅ Opik Active")
        else:
            st.warning("⚠️ Opik Inactive")
    
    with col3:
        st.markdown("**Data Collection**")
        metrics_data = load_metrics_data()
        if metrics_data:
            st.success(f"✅ {len(metrics_data)} records")
        else:
            st.info("ℹ️ No data yet")
    
    # Resource usage
    if resource_usage and any(resource_usage.values()):
        st.subheader("Resource Usage")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_usage = resource_usage.get('cpu_usage', 'N/A')
            st.metric("CPU Usage", cpu_usage)
        
        with col2:
            memory_usage = resource_usage.get('memory_usage', 'N/A')
            st.metric("Memory Usage", memory_usage)
        
        with col3:
            gpu_usage = resource_usage.get('gpu_usage', 'N/A')
            st.metric("GPU Usage", gpu_usage)
    
    # Configuration summary
    st.subheader("Configuration Summary")
    config_info = {
        "Opik Project": get_opik_project_name(),
        "Mode": "Local" if is_local_mode() else "Cloud",
        "Privacy Mode": "Enabled" if observability_config.should_hash_prompts() else "Disabled",
        "Detailed Metrics": "Enabled" if observability_config.get('performance.enable_detailed_metrics') else "Disabled",
        "Trace Retention": f"{observability_config.get('retention.trace_retention_days', 30)} days"
    }
    
    for key, value in config_info.items():
        st.text(f"{key}: {value}")
    
    # Health checks
    st.subheader("Health Checks")
    
    checks = [
        ("Opik Package", OPIK_AVAILABLE),
        ("Legacy Metrics", LEGACY_METRICS_AVAILABLE),
        ("Configuration File", os.path.exists("data/observability_config.json")),
        ("Metrics Data", len(load_metrics_data()) > 0)
    ]
    
    for check_name, status in checks:
        if status:
            st.success(f"✅ {check_name}")
        else:
            st.error(f"❌ {check_name}")


if __name__ == "__main__":
    enhanced_observability_dashboard()
