# model_management.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict, Counter
from ollama_utils import get_available_models, get_ollama_resource_usage, show_model_info

# Define paths for storing model usage and performance data
DATABASE_PATH = "ollama_models.db"
USAGE_DATA_FILE = "model_usage_data.json"
PERFORMANCE_DATA_FILE = "model_performance_data.json"

# Initialize the database if it doesn't exist
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create table for model usage
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        tokens_generated INTEGER,
        response_time REAL,
        operation_type TEXT
    )
    ''')
    
    # Create table for model metadata
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_metadata (
        model_name TEXT PRIMARY KEY,
        size_bytes INTEGER,
        modified_at DATETIME,
        params_billion REAL,
        capabilities TEXT,
        last_used DATETIME
    )
    ''')
    
    # Create table for model performance
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        prompt_text TEXT,
        tokens_per_second REAL,
        latency REAL,
        temperature REAL,
        max_tokens INTEGER
    )
    ''')
    
    # Create table for resource utilization
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resource_utilization (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model_name TEXT,
        cpu_usage REAL,
        memory_usage REAL,
        gpu_usage REAL
    )
    ''')
    
    conn.commit()
    conn.close()

# Function to log model usage
def log_model_usage(model_name, tokens_generated, response_time, operation_type="generate"):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO model_usage (model_name, tokens_generated, response_time, operation_type) VALUES (?, ?, ?, ?)",
        (model_name, tokens_generated, response_time, operation_type)
    )
    
    # Update last_used in metadata
    cursor.execute(
        "UPDATE model_metadata SET last_used = CURRENT_TIMESTAMP WHERE model_name = ?",
        (model_name,)
    )
    
    conn.commit()
    conn.close()

# Function to log model performance
def log_model_performance(model_name, prompt_text, tokens_per_second, latency, temperature, max_tokens):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO model_performance (model_name, prompt_text, tokens_per_second, latency, temperature, max_tokens) VALUES (?, ?, ?, ?, ?, ?)",
        (model_name, prompt_text, tokens_per_second, latency, temperature, max_tokens)
    )
    
    conn.commit()
    conn.close()

# Function to log resource utilization
def log_resource_utilization(model_name=None):
    try:
        usage = get_ollama_resource_usage()
        cpu = float(usage["cpu_usage"].strip('%')) if usage["cpu_usage"] != "N/A" else 0
        memory = float(usage["memory_usage"].strip('%')) if usage["memory_usage"] != "N/A" else 0
        gpu = float(usage["gpu_usage"].strip('%')) if usage["gpu_usage"] != "N/A" else 0
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO resource_utilization (model_name, cpu_usage, memory_usage, gpu_usage) VALUES (?, ?, ?, ?)",
            (model_name, cpu, memory, gpu)
        )
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error logging resource utilization: {str(e)}")

# Function to update model metadata
def update_model_metadata():
    available_models = get_available_models()
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    for model_name in available_models:
        # Skip models with 'embed' in the name
        if 'embed' in model_name.lower():
            continue
            
        # Get model info
        model_info = show_model_info(model_name)
        
        size_bytes = model_info.get('size', 0) if isinstance(model_info, dict) else 0
        modified_at = model_info.get('modified_at', datetime.now().isoformat()) if isinstance(model_info, dict) else datetime.now().isoformat()
        
        # Extract parameter count - this is approximate and based on model name patterns
        params_billion = 0
        if model_info and 'details' in model_info and 'parameter_count' in model_info['details']:
            params_billion = float(model_info['details']['parameter_count']) / 1_000_000_000
        else:
            # Attempt to extract from model name
            for part in model_name.split('-'):
                if part.endswith('b'):
                    try:
                        params_billion = float(part[:-1])
                        break
                    except ValueError:
                        pass
        
        # Extract capabilities
        capabilities = []
        from model_capability_registry import get_model_capabilities
        try:
            model_caps = get_model_capabilities(model_name)
            if model_caps.get("vision", False):
                capabilities.append("Vision")
            if model_caps.get("tools", False):
                capabilities.append("Tools")
            if model_caps.get("embedding", False):
                capabilities.append("Embeddings")
        except Exception:
            pass
        
        capabilities_str = ",".join(capabilities) if capabilities else "Text"
        
        # Check if model already exists in metadata
        cursor.execute("SELECT model_name FROM model_metadata WHERE model_name = ?", (model_name,))
        if cursor.fetchone():
            # Update existing metadata
            cursor.execute(
                "UPDATE model_metadata SET size_bytes = ?, modified_at = ?, params_billion = ?, capabilities = ? WHERE model_name = ?",
                (size_bytes, modified_at, params_billion, capabilities_str, model_name)
            )
        else:
            # Insert new metadata
            cursor.execute(
                "INSERT INTO model_metadata (model_name, size_bytes, modified_at, params_billion, capabilities, last_used) VALUES (?, ?, ?, ?, ?, ?)",
                (model_name, size_bytes, modified_at, params_billion, capabilities_str, None)
            )
    
    conn.commit()
    conn.close()

# Function to get usage statistics for a model
def get_model_usage_stats(model_name=None, days=30):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    date_limit = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    
    if model_name:
        cursor.execute(
            """
            SELECT date(timestamp), SUM(tokens_generated), COUNT(*), AVG(response_time)
            FROM model_usage
            WHERE model_name = ? AND timestamp >= ?
            GROUP BY date(timestamp)
            ORDER BY date(timestamp)
            """,
            (model_name, date_limit)
        )
    else:
        cursor.execute(
            """
            SELECT date(timestamp), SUM(tokens_generated), COUNT(*), AVG(response_time)
            FROM model_usage
            WHERE timestamp >= ?
            GROUP BY date(timestamp)
            ORDER BY date(timestamp)
            """,
            (date_limit,)
        )
    
    results = cursor.fetchall()
    conn.close()
    
    # Prepare data for charts
    dates = [row[0] for row in results]
    tokens = [row[1] for row in results]
    requests = [row[2] for row in results]
    avg_response_times = [row[3] for row in results]
    
    return {
        "dates": dates,
        "tokens": tokens,
        "requests": requests,
        "avg_response_times": avg_response_times
    }

# Function to get performance metrics for a model
def get_model_performance(model_name=None, days=30):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    date_limit = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    
    if model_name:
        cursor.execute(
            """
            SELECT timestamp, tokens_per_second, latency, temperature, max_tokens
            FROM model_performance
            WHERE model_name = ? AND timestamp >= ?
            ORDER BY timestamp
            """,
            (model_name, date_limit)
        )
    else:
        cursor.execute(
            """
            SELECT model_name, AVG(tokens_per_second), AVG(latency)
            FROM model_performance
            WHERE timestamp >= ?
            GROUP BY model_name
            ORDER BY AVG(tokens_per_second) DESC
            """,
            (date_limit,)
        )
    
    results = cursor.fetchall()
    conn.close()
    
    if model_name:
        # Individual model performance over time
        timestamps = [row[0] for row in results]
        tokens_per_second = [row[1] for row in results]
        latencies = [row[2] for row in results]
        temperatures = [row[3] for row in results]
        max_tokens = [row[4] for row in results]
        
        return {
            "timestamps": timestamps,
            "tokens_per_second": tokens_per_second,
            "latencies": latencies,
            "temperatures": temperatures,
            "max_tokens": max_tokens
        }
    else:
        # Comparative model performance
        models = [row[0] for row in results]
        avg_tokens_per_second = [row[1] for row in results]
        avg_latencies = [row[2] for row in results]
        
        return {
            "models": models,
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_latencies": avg_latencies
        }

# Function to get resource utilization for a model
def get_resource_utilization(model_name=None, hours=24):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    time_limit = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
    
    if model_name:
        cursor.execute(
            """
            SELECT timestamp, cpu_usage, memory_usage, gpu_usage
            FROM resource_utilization
            WHERE model_name = ? AND timestamp >= ?
            ORDER BY timestamp
            """,
            (model_name, time_limit)
        )
    else:
        cursor.execute(
            """
            SELECT timestamp, AVG(cpu_usage), AVG(memory_usage), AVG(gpu_usage)
            FROM resource_utilization
            WHERE timestamp >= ?
            GROUP BY strftime('%Y-%m-%d %H', timestamp)
            ORDER BY timestamp
            """,
            (time_limit,)
        )
    
    results = cursor.fetchall()
    conn.close()
    
    timestamps = [row[0] for row in results]
    cpu_usage = [row[1] for row in results]
    memory_usage = [row[2] for row in results]
    gpu_usage = [row[3] for row in results]
    
    return {
        "timestamps": timestamps,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "gpu_usage": gpu_usage
    }

# Function to get models by usage
def get_models_by_usage(days=30):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    date_limit = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute(
        """
        SELECT model_name, COUNT(*) as use_count, SUM(tokens_generated) as total_tokens
        FROM model_usage
        WHERE timestamp >= ?
        GROUP BY model_name
        ORDER BY use_count DESC
        """,
        (date_limit,)
    )
    
    results = cursor.fetchall()
    conn.close()
    
    model_names = [row[0] for row in results]
    use_counts = [row[1] for row in results]
    total_tokens = [row[2] for row in results]
    
    return {
        "model_names": model_names,
        "use_counts": use_counts,
        "total_tokens": total_tokens
    }

# Function to get operation types by model
def get_operation_types(model_name=None, days=30):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    date_limit = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    
    if model_name:
        cursor.execute(
            """
            SELECT operation_type, COUNT(*) as op_count
            FROM model_usage
            WHERE model_name = ? AND timestamp >= ?
            GROUP BY operation_type
            ORDER BY op_count DESC
            """,
            (model_name, date_limit)
        )
    else:
        cursor.execute(
            """
            SELECT operation_type, COUNT(*) as op_count
            FROM model_usage
            WHERE timestamp >= ?
            GROUP BY operation_type
            ORDER BY op_count DESC
            """,
            (date_limit,)
        )
    
    results = cursor.fetchall()
    conn.close()
    
    operation_types = [row[0] for row in results]
    operation_counts = [row[1] for row in results]
    
    return {
        "operation_types": operation_types,
        "operation_counts": operation_counts
    }

# Function to display simulation data if no real data is available
def generate_simulation_data(available_models):
    # Generate simulated data for demo purposes
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_range = [start_date + timedelta(days=i) for i in range(31)]
    
    model_usage_data = {}
    model_performance_data = {}
    
    for model in available_models:
        # Skip embed models
        if 'embed' in model.lower():
            continue
            
        # Generate random usage data
        daily_usage = [int(np.random.normal(5000, 1500)) for _ in range(len(date_range))]
        daily_requests = [int(np.random.normal(50, 15)) for _ in range(len(date_range))]
        avg_response_times = [np.random.normal(0.8, 0.2) for _ in range(len(date_range))]
        
        model_usage_data[model] = {
            "dates": [d.strftime('%Y-%m-%d') for d in date_range],
            "tokens": daily_usage,
            "requests": daily_requests,
            "avg_response_times": avg_response_times
        }
        
        # Generate random performance data
        tokens_per_second = np.random.normal(30, 10)
        latency = np.random.normal(1.2, 0.3)
        
        model_performance_data[model] = {
            "avg_tokens_per_second": tokens_per_second,
            "avg_latency": latency
        }
    
    # Generate random resource utilization data
    time_points = [end_date - timedelta(hours=i) for i in range(24, 0, -1)]
    cpu_usage = [min(95, max(5, np.random.normal(40, 15))) for _ in range(len(time_points))]
    memory_usage = [min(95, max(5, np.random.normal(60, 10))) for _ in range(len(time_points))]
    gpu_usage = [min(95, max(5, np.random.normal(70, 20))) for _ in range(len(time_points))]
    
    resource_data = {
        "timestamps": [t.strftime('%Y-%m-%d %H:%M:%S') for t in time_points],
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "gpu_usage": gpu_usage
    }
    
    return model_usage_data, model_performance_data, resource_data

# Main dashboard function
def model_management_dashboard():
    st.title("📊 Model Management Dashboard")
    
    # Initialize database if it doesn't exist
    init_db()
    
    # Update model metadata
    with st.spinner("Updating model metadata..."):
        update_model_metadata()
    
    # Get available models
    available_models = get_available_models()
    
    # Setup the dashboard
    st.markdown("### Model Usage and Performance Analytics")
    
    # Time period filter
    col1, col2 = st.columns(2)
    with col1:
        time_period = st.selectbox(
            "Time Period:",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            index=1
        )
    
    with col2:
        model_filter = st.selectbox(
            "Filter by Model:",
            ["All Models"] + available_models,
            index=0
        )
    
    # Convert time_period to days
    days_mapping = {
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "All Time": 365  # Using 1 year as "All Time" for simplicity
    }
    days = days_mapping[time_period]
    
    model_name = None if model_filter == "All Models" else model_filter
    
    # Get data from database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Check if we have real data
    cursor.execute("SELECT COUNT(*) FROM model_usage")
    has_usage_data = cursor.fetchone()[0] > 0
    
    # Check if we have performance data
    cursor.execute("SELECT COUNT(*) FROM model_performance")
    has_performance_data = cursor.fetchone()[0] > 0
    
    # Check if we have resource data
    cursor.execute("SELECT COUNT(*) FROM resource_utilization")
    has_resource_data = cursor.fetchone()[0] > 0
    
    conn.close()
    
    # If no data, generate simulation data
    if not has_usage_data and not has_performance_data:
        st.info("No usage data found in the database. Displaying simulated data for demonstration purposes.")
        model_usage_data, model_performance_data, resource_data = generate_simulation_data(available_models)
        
        if model_name:
            usage_data = model_usage_data.get(model_name, {})
            performance_data = {
                "models": [model_name],
                "avg_tokens_per_second": [model_performance_data.get(model_name, {}).get("avg_tokens_per_second", 0)],
                "avg_latencies": [model_performance_data.get(model_name, {}).get("avg_latency", 0)]
            }
        else:
            # Combine all model data
            usage_data = {
                "dates": model_usage_data[available_models[0]]["dates"],
                "tokens": [0] * len(model_usage_data[available_models[0]]["dates"]),
                "requests": [0] * len(model_usage_data[available_models[0]]["dates"]),
                "avg_response_times": [0] * len(model_usage_data[available_models[0]]["dates"])
            }
            
            for model in available_models:
                if 'embed' in model.lower() or model not in model_usage_data:
                    continue
                
                for i in range(len(usage_data["dates"])):
                    usage_data["tokens"][i] += model_usage_data[model]["tokens"][i]
                    usage_data["requests"][i] += model_usage_data[model]["requests"][i]
                    usage_data["avg_response_times"][i] += model_usage_data[model]["avg_response_times"][i] / len(available_models)
            
            performance_data = {
                "models": available_models,
                "avg_tokens_per_second": [model_performance_data.get(model, {}).get("avg_tokens_per_second", 0) for model in available_models],
                "avg_latencies": [model_performance_data.get(model, {}).get("avg_latency", 0) for model in available_models]
            }
    else:
        # Get real data from database
        usage_data = get_model_usage_stats(model_name, days)
        performance_data = get_model_performance(model_name, days)
        resource_data = get_resource_utilization(model_name, hours=days*24)
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance Metrics", "Resource Utilization", "Metadata"])
    
    with tab1:
        # Usage statistics 
        st.subheader("📈 Usage Statistics")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate summary metrics
        total_tokens = sum(usage_data.get("tokens", [0]))
        total_requests = sum(usage_data.get("requests", [0]))
        avg_response_time = np.mean(usage_data.get("avg_response_times", [0])) if usage_data.get("avg_response_times") else 0
        
        with col1:
            st.metric("Total Tokens Generated", f"{total_tokens:,}")
        
        with col2:
            st.metric("Total Requests", f"{total_requests:,}")
        
        with col3:
            st.metric("Avg. Response Time", f"{avg_response_time:.2f}s")
        
        with col4:
            # Get most used model
            if not model_name and has_usage_data:
                usage_by_model = get_models_by_usage(days)
                most_used_model = usage_by_model["model_names"][0] if usage_by_model["model_names"] else "N/A"
                st.metric("Most Used Model", most_used_model)
            else:
                # If model filter is active or no data
                st.metric("Active Models", len([m for m in available_models if 'embed' not in m.lower()]))
        
        # Usage over time
        st.subheader("Usage Over Time")
        
        if usage_data.get("dates"):
            # Create dataframe for combined chart
            usage_df = pd.DataFrame({
                "Date": usage_data["dates"],
                "Tokens Generated": usage_data["tokens"],
                "Requests": usage_data["requests"]
            })
            
            # Normalize data for dual axis
            max_tokens = max(usage_data["tokens"]) if usage_data["tokens"] else 1
            max_requests = max(usage_data["requests"]) if usage_data["requests"] else 1
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add tokens line
            fig.add_trace(go.Scatter(
                x=usage_df["Date"],
                y=usage_df["Tokens Generated"],
                name="Tokens Generated",
                line=dict(color="#1f77b4", width=2)
            ))
            
            # Add requests line on secondary axis
            fig.add_trace(go.Scatter(
                x=usage_df["Date"],
                y=usage_df["Requests"],
                name="Requests",
                line=dict(color="#ff7f0e", width=2),
                yaxis="y2"
            ))
            
            # Update layout for dual y-axis
            fig.update_layout(
                title="Model Usage Trends",
                xaxis=dict(title="Date"),
                yaxis=dict(
                    title="Tokens Generated",
                    titlefont=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")
                ),
                yaxis2=dict(
                    title="Requests",
                    titlefont=dict(color="#ff7f0e"),
                    tickfont=dict(color="#ff7f0e"),
                    overlaying="y",
                    side="right"
                ),
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No usage data available for the selected time period.")
        
        # Model Usage Distribution
        if not model_name and has_usage_data:
            st.subheader("Model Usage Distribution")
            
            usage_by_model = get_models_by_usage(days)
            
            if usage_by_model["model_names"]:
                usage_df = pd.DataFrame({
                    "Model": usage_by_model["model_names"],
                    "Requests": usage_by_model["use_counts"],
                    "Tokens": usage_by_model["total_tokens"]
                })
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Requests by model pie chart
                    fig1 = px.pie(
                        usage_df,
                        values="Requests",
                        names="Model",
                        title="Requests by Model"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Tokens by model pie chart
                    fig2 = px.pie(
                        usage_df,
                        values="Tokens",
                        names="Model",
                        title="Tokens Generated by Model"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No usage distribution data available for the selected time period.")
        
        # Operation Types
        st.subheader("Operation Types")
        
        operation_data = get_operation_types(model_name, days)
        
        if operation_data["operation_types"]:
            op_df = pd.DataFrame({
                "Operation": operation_data["operation_types"],
                "Count": operation_data["operation_counts"]
            })
            
            fig = px.bar(
                op_df,
                x="Operation",
                y="Count",
                title="Operations by Type",
                color="Operation"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No operation type data available for the selected time period.")
    
    with tab2:
        # Performance metrics tab
        st.subheader("⚡ Performance Metrics")
        
        if not model_name:
            # Comparative performance across models
            if has_performance_data or performance_data.get("models"):
                perf_df = pd.DataFrame({
                    "Model": performance_data["models"],
                    "Tokens/Second": performance_data["avg_tokens_per_second"],
                    "Latency (s)": performance_data["avg_latencies"]
                })
                
                # Sort by tokens/second descending
                perf_df = perf_df.sort_values("Tokens/Second", ascending=False)
                
                # Two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tokens per second by model
                    fig1 = px.bar(
                        perf_df,
                        x="Model",
                        y="Tokens/Second",
                        title="Average Tokens per Second by Model",
                        color="Model"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Latency by model
                    fig2 = px.bar(
                        perf_df,
                        x="Model",
                        y="Latency (s)",
                        title="Average Latency by Model",
                        color="Model"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Display performance data table
                st.dataframe(perf_df, use_container_width=True)
            else:
                st.info("No performance data available for the selected time period.")
        else:
            # Individual model performance over time
            if has_performance_data and performance_data.get("timestamps"):
                # Create dataframe for time series
                perf_time_df = pd.DataFrame({
                    "Timestamp": performance_data["timestamps"],
                    "Tokens/Second": performance_data["tokens_per_second"],
                    "Latency (s)": performance_data["latencies"],
                    "Temperature": performance_data["temperatures"],
                    "Max Tokens": performance_data["max_tokens"]
                })
                
                # Plot tokens per second over time
                fig1 = px.line(
                    perf_time_df,
                    x="Timestamp",
                    y="Tokens/Second",
                    title=f"Tokens per Second Over Time for {model_name}"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Plot latency over time
                fig2 = px.line(
                    perf_time_df,
                    x="Timestamp",
                    y="Latency (s)",
                    title=f"Response Latency Over Time for {model_name}"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show correlation between temperature and performance
                fig3 = px.scatter(
                    perf_time_df,
                    x="Temperature",
                    y="Tokens/Second",
                    title="Temperature vs Performance",
                    trendline="ols"
                )
                st.plotly_chart(fig3, use_container_width=True)
            elif performance_data:
                # Show simulated data for demo
                st.metric("Average Tokens per Second", f"{performance_data['avg_tokens_per_second'][0]:.2f}")
                st.metric("Average Latency", f"{performance_data['avg_latencies'][0]:.2f}s")
                st.info("Detailed performance history not available for this model. Only averages are shown.")
            else:
                st.info("No performance data available for this model in the selected time period.")
    
    with tab3:
        # Resource utilization tab
        st.subheader("🖥️ Resource Utilization")
        
        if has_resource_data or resource_data:
            # Create dataframe for resource utilization
            resource_df = pd.DataFrame({
                "Timestamp": resource_data["timestamps"],
                "CPU Usage (%)": resource_data["cpu_usage"],
                "Memory Usage (%)": resource_data["memory_usage"],
                "GPU Usage (%)": resource_data["gpu_usage"]
            })
            
            # Plot resource utilization over time
            fig = go.Figure()
            
            # Add CPU usage line
            fig.add_trace(go.Scatter(
                x=resource_df["Timestamp"],
                y=resource_df["CPU Usage (%)"],
                name="CPU Usage",
                line=dict(color="#1f77b4", width=2)
            ))
            
            # Add Memory usage line
            fig.add_trace(go.Scatter(
                x=resource_df["Timestamp"],
                y=resource_df["Memory Usage (%)"],
                name="Memory Usage",
                line=dict(color="#ff7f0e", width=2)
            ))
            
            # Add GPU usage line if available
            if not all(gpu == 0 for gpu in resource_data["gpu_usage"]):
                fig.add_trace(go.Scatter(
                    x=resource_df["Timestamp"],
                    y=resource_df["GPU Usage (%)"],
                    name="GPU Usage",
                    line=dict(color="#2ca02c", width=2)
                ))
            
            # Update layout
            fig.update_layout(
                title="Resource Utilization Over Time",
                xaxis=dict(title="Time"),
                yaxis=dict(title="Usage (%)"),
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current resource usage
            st.subheader("Current Resource Usage")
            current_usage = get_ollama_resource_usage()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CPU Usage", current_usage["cpu_usage"])
            
            with col2:
                st.metric("Memory Usage", current_usage["memory_usage"])
            
            with col3:
                if current_usage["gpu_usage"] != "N/A":
                    st.metric("GPU Usage", current_usage["gpu_usage"])
                else:
                    st.metric("GPU Usage", "N/A")
        else:
            st.info("No resource utilization data available for the selected time period.")
            
            # Still show current resource usage even if no history
            st.subheader("Current Resource Usage")
            current_usage = get_ollama_resource_usage()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CPU Usage", current_usage["cpu_usage"])
            
            with col2:
                st.metric("Memory Usage", current_usage["memory_usage"])
            
            with col3:
                if current_usage["gpu_usage"] != "N/A":
                    st.metric("GPU Usage", current_usage["gpu_usage"])
                else:
                    st.metric("GPU Usage", "N/A")
    
    with tab4:
        # Model metadata tab
        st.subheader("📋 Model Metadata")
        
        # Get model metadata
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT model_name, size_bytes, modified_at, params_billion, capabilities, last_used
            FROM model_metadata
            ORDER BY
                CASE
                    WHEN last_used IS NULL THEN 1
                    ELSE 0
                END,
                last_used DESC
            """
        )
        
        metadata_results = cursor.fetchall()
        conn.close()
        
        if metadata_results:
            # Create dataframe for metadata
            metadata_df = pd.DataFrame(metadata_results, columns=[
                "Model", "Size (bytes)", "Modified", "Parameters (B)", "Capabilities", "Last Used"
            ])
            
            # Convert bytes to GB
            metadata_df["Size (GB)"] = metadata_df["Size (bytes)"] / (1024**3)
            metadata_df = metadata_df.drop(columns=["Size (bytes)"])
            
            # Format "Last Used" column
            metadata_df["Last Used"] = metadata_df["Last Used"].apply(
                lambda x: "Never" if x is None else x
            )
            
            # Reorder columns
            metadata_df = metadata_df[["Model", "Size (GB)", "Parameters (B)", "Capabilities", "Modified", "Last Used"]]
            
            # Display metadata table
            st.dataframe(metadata_df, use_container_width=True)
            
            # Model size distribution
            st.subheader("Model Size Distribution")
            
            fig = px.bar(
                metadata_df,
                x="Model",
                y="Size (GB)",
                title="Model Size Distribution (GB)",
                color="Model"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model parameter count distribution
            st.subheader("Model Parameter Count Distribution")
            
            fig = px.bar(
                metadata_df,
                x="Model",
                y="Parameters (B)",
                title="Model Parameter Count (Billions)",
                color="Model"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Capabilities distribution
            st.subheader("Model Capabilities")
            
            # Extract individual capabilities
            capabilities = []
            for cap_str in metadata_df["Capabilities"]:
                caps = cap_str.split(",")
                for cap in caps:
                    if cap:
                        capabilities.append(cap)
            
            capability_counts = Counter(capabilities)
            
            # Create dataframe for capability distribution
            cap_df = pd.DataFrame({
                "Capability": list(capability_counts.keys()),
                "Count": list(capability_counts.values())
            })
            
            fig = px.pie(
                cap_df,
                values="Count",
                names="Capability",
                title="Model Capabilities Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model metadata available. Try updating the model metadata.")
    
    # Add action buttons
    st.subheader("Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
    
    with col2:
        if st.button("📊 Log Current Resource Usage"):
            with st.spinner("Logging resource usage..."):
                log_resource_utilization(model_name)
                st.success("Resource usage logged successfully!")
    
    with col3:
        if st.button("💾 Update Model Metadata"):
            with st.spinner("Updating model metadata..."):
                update_model_metadata()
                st.success("Model metadata updated successfully!")
    
    # Add instructions for integration
    st.subheader("Integration Instructions")
    
    st.markdown("""
    To track model usage in your application, add the following code to your model calls:
    
    ```python
    from model_management import log_model_usage, log_model_performance
    
    # After calling a model
    log_model_usage(
        model_name="your_model_name",
        tokens_generated=response_tokens,  # number of tokens generated
        response_time=elapsed_time,  # time taken in seconds
        operation_type="generate"  # or "embed", "chat", etc.
    )
    
    # For performance metrics
    log_model_performance(
        model_name="your_model_name",
        prompt_text="summary of prompt",  # or full prompt text
        tokens_per_second=tokens_per_second,
        latency=latency,
        temperature=temperature,
        max_tokens=max_tokens
    )
    ```
    """)
    
    # Display data sources
    st.caption("Data sources: model_usage, model_performance, resource_utilization, and model_metadata tables in the SQLite database.")

# Run the dashboard when this file is executed directly
if __name__ == "__main__":
    model_management_dashboard()