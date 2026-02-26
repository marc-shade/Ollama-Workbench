import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime

def performance_metrics_interface():
    st.title("Model Performance Metrics")
    
    # Tabs for different metric views
    tab1, tab2, tab3 = st.tabs(["Response Time", "Token Usage", "Model Comparison"])
    
    with tab1:
        display_response_time_metrics()
    
    with tab2:
        display_token_usage_metrics()
    
    with tab3:
        display_model_comparison()

def display_response_time_metrics():
    st.subheader("Response Time Metrics")
    
    # Check if metrics file exists and load data
    metrics_data = load_metrics_data()
    if not metrics_data:
        st.info("No performance metrics data available yet. Start using the chat to generate metrics.")
        return
    
    # Create DataFrame for response times
    response_times = []
    for entry in metrics_data:
        if "response_time" in entry:
            response_times.append({
                "timestamp": entry.get("timestamp", "Unknown"),
                "model": entry.get("model", "Unknown"),
                "response_time": entry.get("response_time", 0),
                "prompt_length": entry.get("prompt_length", 0)
            })
    
    if not response_times:
        st.info("No response time data available yet.")
        return
    
    df_response = pd.DataFrame(response_times)
    
    # Convert timestamp to datetime with error handling
    try:
        df_response["timestamp"] = pd.to_datetime(df_response["timestamp"], errors='coerce')
        # Remove rows with invalid timestamps
        df_response = df_response.dropna(subset=['timestamp'])
        df_response = df_response.sort_values("timestamp")
    except Exception:
        # If timestamp parsing fails completely, use index as fallback
        df_response["timestamp"] = pd.to_datetime(df_response.index, unit='s', origin='unix')
    
    # Response time over time
    st.subheader("Response Time Over Time")
    fig = px.line(df_response, x="timestamp", y="response_time", color="model",
                  labels={"response_time": "Response Time (s)", "timestamp": "Time"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Average response time by model
    st.subheader("Average Response Time by Model")
    avg_by_model = df_response.groupby("model")["response_time"].mean().reset_index()
    fig = px.bar(avg_by_model, x="model", y="response_time", 
                 labels={"response_time": "Average Response Time (s)", "model": "Model"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Response time vs prompt length
    st.subheader("Response Time vs Prompt Length")
    fig = px.scatter(df_response, x="prompt_length", y="response_time", color="model",
                     labels={"response_time": "Response Time (s)", "prompt_length": "Prompt Length (chars)"})
    st.plotly_chart(fig, use_container_width=True)

def display_token_usage_metrics():
    st.subheader("Token Usage Metrics")
    
    # Check if metrics file exists and load data
    metrics_data = load_metrics_data()
    if not metrics_data:
        st.info("No performance metrics data available yet. Start using the chat to generate metrics.")
        return
    
    # Create DataFrame for token usage
    token_usage = []
    for entry in metrics_data:
        if "input_tokens" in entry or "output_tokens" in entry:
            token_usage.append({
                "timestamp": entry.get("timestamp", "Unknown"),
                "model": entry.get("model", "Unknown"),
                "input_tokens": entry.get("input_tokens", 0),
                "output_tokens": entry.get("output_tokens", 0),
                "total_tokens": entry.get("input_tokens", 0) + entry.get("output_tokens", 0)
            })
    
    if not token_usage:
        st.info("No token usage data available yet.")
        return
    
    df_tokens = pd.DataFrame(token_usage)
    
    # Convert timestamp to datetime with error handling
    try:
        df_tokens["timestamp"] = pd.to_datetime(df_tokens["timestamp"], errors='coerce')
        # Remove rows with invalid timestamps
        df_tokens = df_tokens.dropna(subset=['timestamp'])
        df_tokens = df_tokens.sort_values("timestamp")
    except Exception:
        # If timestamp parsing fails completely, use index as fallback
        df_tokens["timestamp"] = pd.to_datetime(df_tokens.index, unit='s', origin='unix')
    
    # Token usage over time
    st.subheader("Token Usage Over Time")
    fig = px.line(df_tokens, x="timestamp", y="total_tokens", color="model",
                  labels={"total_tokens": "Total Tokens", "timestamp": "Time"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Input vs Output tokens
    st.subheader("Input vs Output Tokens by Model")
    models = df_tokens["model"].unique()
    
    input_avg = []
    output_avg = []
    
    for model in models:
        model_data = df_tokens[df_tokens["model"] == model]
        input_avg.append(model_data["input_tokens"].mean())
        output_avg.append(model_data["output_tokens"].mean())
    
    fig = go.Figure(data=[
        go.Bar(name="Input Tokens", x=models, y=input_avg),
        go.Bar(name="Output Tokens", x=models, y=output_avg)
    ])
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    
    # Total token usage by model
    st.subheader("Total Token Usage by Model")
    total_by_model = df_tokens.groupby("model")[["input_tokens", "output_tokens", "total_tokens"]].sum().reset_index()
    fig = px.pie(total_by_model, values="total_tokens", names="model", 
                 title="Proportion of Total Tokens Used by Model")
    st.plotly_chart(fig, use_container_width=True)

def display_model_comparison():
    st.subheader("Model Comparison Dashboard")
    
    # Check if metrics file exists and load data
    metrics_data = load_metrics_data()
    if not metrics_data:
        st.info("No performance metrics data available yet. Start using the chat to generate metrics.")
        return
    
    # Create DataFrame for model comparison
    comparison_data = []
    for entry in metrics_data:
        # Only include entries with all the metrics we need for comparison
        if all(key in entry for key in ["model", "response_time", "input_tokens", "output_tokens"]):
            comparison_data.append({
                "model": entry.get("model", "Unknown"),
                "response_time": entry.get("response_time", 0),
                "input_tokens": entry.get("input_tokens", 0),
                "output_tokens": entry.get("output_tokens", 0),
                "tokens_per_second": entry.get("output_tokens", 0) / max(entry.get("response_time", 1), 0.001),
                "timestamp": entry.get("timestamp", "Unknown")
            })
    
    if not comparison_data:
        st.info("Not enough data for model comparison yet.")
        return
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Average metrics by model
    avg_metrics = df_comparison.groupby("model").agg({
        "response_time": "mean",
        "input_tokens": "mean",
        "output_tokens": "mean",
        "tokens_per_second": "mean"
    }).reset_index()
    
    # Display comparison table
    st.subheader("Average Performance Metrics by Model")
    st.dataframe(avg_metrics.round(2))
    
    # Tokens per second comparison (throughput)
    st.subheader("Model Throughput (Tokens per Second)")
    fig = px.bar(avg_metrics, x="model", y="tokens_per_second",
                labels={"tokens_per_second": "Output Tokens per Second", "model": "Model"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for model comparison
    st.subheader("Model Performance Radar Chart")
    
    # Normalize metrics for radar chart
    radar_metrics = avg_metrics.copy()
    
    # For response time, lower is better so we invert it
    max_response_time = radar_metrics["response_time"].max()
    radar_metrics["response_time_norm"] = (max_response_time - radar_metrics["response_time"]) / max_response_time
    
    # For other metrics, higher is better
    radar_metrics["input_tokens_norm"] = radar_metrics["input_tokens"] / radar_metrics["input_tokens"].max()
    radar_metrics["output_tokens_norm"] = radar_metrics["output_tokens"] / radar_metrics["output_tokens"].max()
    radar_metrics["tokens_per_second_norm"] = radar_metrics["tokens_per_second"] / radar_metrics["tokens_per_second"].max()
    
    # Create radar chart
    categories = ["Speed", "Input Capacity", "Output Size", "Throughput"]
    
    fig = go.Figure()
    for i, model in enumerate(radar_metrics["model"]):
        values = [
            radar_metrics.iloc[i]["response_time_norm"],
            radar_metrics.iloc[i]["input_tokens_norm"],
            radar_metrics.iloc[i]["output_tokens_norm"],
            radar_metrics.iloc[i]["tokens_per_second_norm"]
        ]
        values.append(values[0])  # Close the loop
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],  # Close the loop
            fill="toself",
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def load_metrics_data():
    """Load metrics data from the metrics file"""
    metrics_file = os.path.join("data", "performance_metrics.json")
    
    if not os.path.exists(metrics_file):
        return []
    
    try:
        with open(metrics_file, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics data: {e}")
        return []

def record_metrics(model, response_time, prompt_length=None, input_tokens=None, output_tokens=None):
    """Record performance metrics to the metrics file"""
    metrics_dir = "data"
    metrics_file = os.path.join(metrics_dir, "performance_metrics.json")
    
    # Create directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load existing metrics
    metrics_data = []
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
        except:
            # If file exists but can't be loaded, start with empty list
            metrics_data = []
    
    # Add new metrics entry
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "response_time": response_time
    }
    
    if prompt_length is not None:
        new_entry["prompt_length"] = prompt_length
    
    if input_tokens is not None:
        new_entry["input_tokens"] = input_tokens
    
    if output_tokens is not None:
        new_entry["output_tokens"] = output_tokens
    
    metrics_data.append(new_entry)
    
    # Save updated metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)

if __name__ == "__main__":
    # For testing the component in isolation
    performance_metrics_interface()