import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime
import os
import glob
import re
from typing import Dict, Any, List, Optional, Tuple, Union

# Define colors for different result types
RESULT_COLORS = {
    "pass": "#76b852",  # Green
    "fail": "#e74c3c",  # Red
    "warning": "#f39c12",  # Yellow
    "info": "#3498db",   # Blue
    "skip": "#95a5a6"    # Gray
}

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"

def get_test_results(test_dir: str = None) -> List[Dict[str, Any]]:
    """
    Get test results from files in the specified directory.
    
    Args:
        test_dir: Directory containing test result files
        
    Returns:
        List[Dict[str, Any]]: List of test results
    """
    if test_dir is None:
        test_dir = os.path.join(os.getcwd(), "test_results")
    
    # Create directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    
    results = []
    
    # Find JSON files in the test directory
    for file_path in glob.glob(os.path.join(test_dir, "*.json")):
        try:
            with open(file_path, "r") as f:
                result = json.load(f)
                
                # Add file info
                file_name = os.path.basename(file_path)
                file_stats = os.stat(file_path)
                
                result["file_name"] = file_name
                result["file_path"] = file_path
                result["file_size"] = file_stats.st_size
                result["file_mtime"] = file_stats.st_mtime
                
                results.append(result)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    # Sort by modification time (newest first)
    results.sort(key=lambda x: x.get("file_mtime", 0), reverse=True)
    
    return results

def save_test_result(result: Dict[str, Any], test_dir: str = None) -> str:
    """
    Save test result to a file.
    
    Args:
        result: Test result to save
        test_dir: Directory to save the file in
        
    Returns:
        str: Path to the saved file
    """
    if test_dir is None:
        test_dir = os.path.join(os.getcwd(), "test_results")
    
    # Create directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate a file name based on test type and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_type = result.get("test_type", "unknown").lower().replace(" ", "_")
    file_name = f"{test_type}_{timestamp}.json"
    file_path = os.path.join(test_dir, file_name)
    
    # Save the result
    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)
    
    return file_path

def create_summary_chart(results: List[Dict[str, Any]], chart_type: str = "bar") -> go.Figure:
    """
    Create a summary chart of test results.
    
    Args:
        results: List of test results
        chart_type: Type of chart to create ("bar", "pie", "scatter", etc.)
        
    Returns:
        go.Figure: Plotly figure
    """
    # Extract test types and results
    summary_data = {}
    
    for result in results:
        test_type = result.get("test_type", "Unknown")
        status = result.get("status", "Unknown")
        
        if test_type not in summary_data:
            summary_data[test_type] = {"pass": 0, "fail": 0, "skip": 0, "warning": 0, "info": 0}
        
        if status.lower() in ["pass", "success", "passed", "ok"]:
            summary_data[test_type]["pass"] += 1
        elif status.lower() in ["fail", "failure", "failed", "error"]:
            summary_data[test_type]["fail"] += 1
        elif status.lower() in ["skip", "skipped", "ignored"]:
            summary_data[test_type]["skip"] += 1
        elif status.lower() in ["warning", "warn"]:
            summary_data[test_type]["warning"] += 1
        else:
            summary_data[test_type]["info"] += 1
    
    # Prepare data for chart
    chart_data = []
    
    for test_type, counts in summary_data.items():
        for status, count in counts.items():
            if count > 0:
                chart_data.append({
                    "Test Type": test_type,
                    "Status": status.capitalize(),
                    "Count": count,
                    "Color": RESULT_COLORS.get(status.lower(), "#95a5a6")
                })
    
    # Create DataFrame
    df = pd.DataFrame(chart_data)
    
    # Create chart
    if chart_type == "bar":
        fig = px.bar(
            df,
            x="Test Type",
            y="Count",
            color="Status",
            color_discrete_map={
                "Pass": RESULT_COLORS["pass"],
                "Fail": RESULT_COLORS["fail"],
                "Warning": RESULT_COLORS["warning"],
                "Info": RESULT_COLORS["info"],
                "Skip": RESULT_COLORS["skip"]
            },
            title="Test Results Summary",
            labels={"Test Type": "Test Type", "Count": "Number of Tests"}
        )
    elif chart_type == "pie":
        fig = px.pie(
            df,
            values="Count",
            names="Status",
            color="Status",
            color_discrete_map={
                "Pass": RESULT_COLORS["pass"],
                "Fail": RESULT_COLORS["fail"],
                "Warning": RESULT_COLORS["warning"],
                "Info": RESULT_COLORS["info"],
                "Skip": RESULT_COLORS["skip"]
            },
            title="Test Results by Status"
        )
    else:
        # Default to bar chart
        fig = px.bar(
            df,
            x="Test Type",
            y="Count",
            color="Status",
            color_discrete_map={
                "Pass": RESULT_COLORS["pass"],
                "Fail": RESULT_COLORS["fail"],
                "Warning": RESULT_COLORS["warning"],
                "Info": RESULT_COLORS["info"],
                "Skip": RESULT_COLORS["skip"]
            },
            title="Test Results Summary",
            labels={"Test Type": "Test Type", "Count": "Number of Tests"}
        )
    
    # Update layout
    fig.update_layout(
        legend_title="Status",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_performance_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a performance chart of test results.
    
    Args:
        results: List of test results
        
    Returns:
        go.Figure: Plotly figure
    """
    # Extract model names and durations
    performance_data = []
    
    for result in results:
        # Skip results without duration or model name
        if "duration" not in result or "model" not in result:
            continue
        
        test_type = result.get("test_type", "Unknown")
        model = result.get("model", "Unknown")
        duration = result.get("duration", 0)
        status = result.get("status", "Unknown").lower()
        
        # Map status to color
        if status in ["pass", "success", "passed", "ok"]:
            color = RESULT_COLORS["pass"]
        elif status in ["fail", "failure", "failed", "error"]:
            color = RESULT_COLORS["fail"]
        elif status in ["skip", "skipped", "ignored"]:
            color = RESULT_COLORS["skip"]
        elif status in ["warning", "warn"]:
            color = RESULT_COLORS["warning"]
        else:
            color = RESULT_COLORS["info"]
        
        performance_data.append({
            "Test Type": test_type,
            "Model": model,
            "Duration": duration,
            "Status": status.capitalize(),
            "Color": color
        })
    
    # Create DataFrame
    df = pd.DataFrame(performance_data)
    
    if df.empty:
        # Create empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No performance data available",
            xaxis=dict(title="Model"),
            yaxis=dict(title="Duration (s)")
        )
        return fig
    
    # Calculate average duration by model and test type
    avg_durations = df.groupby(["Test Type", "Model"])["Duration"].mean().reset_index()
    
    # Create chart
    fig = px.bar(
        avg_durations,
        x="Model",
        y="Duration",
        color="Test Type",
        title="Average Test Duration by Model",
        labels={"Model": "Model Name", "Duration": "Average Duration (s)"}
    )
    
    # Update layout
    fig.update_layout(
        legend_title="Test Type",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_model_comparison_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a model comparison chart.
    
    Args:
        results: List of test results
        
    Returns:
        go.Figure: Plotly figure
    """
    # Extract model names and scores
    comparison_data = []
    
    for result in results:
        # Skip results without score or model name
        if "score" not in result or "model" not in result:
            continue
        
        test_type = result.get("test_type", "Unknown")
        model = result.get("model", "Unknown")
        score = result.get("score", 0)
        
        comparison_data.append({
            "Test Type": test_type,
            "Model": model,
            "Score": score
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    if df.empty:
        # Create empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No model comparison data available",
            xaxis=dict(title="Model"),
            yaxis=dict(title="Score")
        )
        return fig
    
    # Calculate average score by model and test type
    avg_scores = df.groupby(["Test Type", "Model"])["Score"].mean().reset_index()
    
    # Create chart
    fig = px.bar(
        avg_scores,
        x="Model",
        y="Score",
        color="Test Type",
        title="Average Test Scores by Model",
        labels={"Model": "Model Name", "Score": "Average Score"}
    )
    
    # Update layout
    fig.update_layout(
        legend_title="Test Type",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_timeline_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a timeline chart of test results.
    
    Args:
        results: List of test results
        
    Returns:
        go.Figure: Plotly figure
    """
    # Extract timestamps and test types
    timeline_data = []
    
    for result in results:
        # Use either timestamp from result or file modification time
        timestamp = result.get("timestamp", result.get("file_mtime", 0))
        
        # Convert timestamp to datetime
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    try:
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        timestamp = datetime.fromtimestamp(result.get("file_mtime", 0))
        
        test_type = result.get("test_type", "Unknown")
        status = result.get("status", "Unknown").lower()
        
        # Map status to color
        if status in ["pass", "success", "passed", "ok"]:
            color = RESULT_COLORS["pass"]
        elif status in ["fail", "failure", "failed", "error"]:
            color = RESULT_COLORS["fail"]
        elif status in ["skip", "skipped", "ignored"]:
            color = RESULT_COLORS["skip"]
        elif status in ["warning", "warn"]:
            color = RESULT_COLORS["warning"]
        else:
            color = RESULT_COLORS["info"]
        
        timeline_data.append({
            "Test Type": test_type,
            "Timestamp": timestamp,
            "Status": status.capitalize(),
            "Color": color
        })
    
    # Create DataFrame
    df = pd.DataFrame(timeline_data)
    
    if df.empty:
        # Create empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No timeline data available",
            xaxis=dict(title="Date/Time"),
            yaxis=dict(title="Test Type")
        )
        return fig
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x="Timestamp",
        y="Test Type",
        color="Status",
        color_discrete_map={
            "Pass": RESULT_COLORS["pass"],
            "Fail": RESULT_COLORS["fail"],
            "Warning": RESULT_COLORS["warning"],
            "Info": RESULT_COLORS["info"],
            "Skip": RESULT_COLORS["skip"]
        },
        title="Test Results Timeline",
        labels={"Timestamp": "Date/Time", "Test Type": "Test Type"}
    )
    
    # Update layout
    fig.update_layout(
        legend_title="Status",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def generate_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate a summary table of test results.
    
    Args:
        results: List of test results
        
    Returns:
        pd.DataFrame: Summary table
    """
    summary_data = []
    
    for result in results:
        test_type = result.get("test_type", "Unknown")
        model = result.get("model", "Unknown")
        status = result.get("status", "Unknown")
        timestamp = result.get("timestamp", result.get("file_mtime", 0))
        
        # Convert timestamp to datetime
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    try:
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        timestamp = datetime.fromtimestamp(result.get("file_mtime", 0))
        
        duration = result.get("duration", 0)
        score = result.get("score", None)
        
        summary_data.append({
            "Test Type": test_type,
            "Model": model,
            "Status": status,
            "Timestamp": timestamp,
            "Duration": duration,
            "Score": score
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Format duration
    if "Duration" in df.columns:
        df["Duration"] = df["Duration"].apply(lambda x: format_duration(x) if x else "N/A")
    
    # Format timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = df["Timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if hasattr(x, "strftime") else "N/A")
    
    # Format score
    if "Score" in df.columns:
        df["Score"] = df["Score"].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
    
    return df

def test_visualization_ui():
    """Main UI for test visualization."""
    st.title("📊 Test Visualization")
    st.write("Visualize and analyze test results")
    
    # Get test results
    results = get_test_results()
    
    if not results:
        st.warning("No test results found. Run some tests first.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Summary", "Model Comparison", "Timeline", "Details"
    ])
    
    with tab1:
        st.subheader("Test Results Summary")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        total_tests = len(results)
        pass_count = sum(1 for r in results if r.get("status", "").lower() in ["pass", "success", "passed", "ok"])
        fail_count = sum(1 for r in results if r.get("status", "").lower() in ["fail", "failure", "failed", "error"])
        skip_count = sum(1 for r in results if r.get("status", "").lower() in ["skip", "skipped", "ignored"])
        
        with col1:
            st.metric("Total Tests", total_tests)
        
        with col2:
            st.metric("Passed", pass_count, f"{pass_count / total_tests * 100:.1f}%" if total_tests > 0 else "0%")
        
        with col3:
            st.metric("Failed", fail_count, f"{fail_count / total_tests * 100:.1f}%" if total_tests > 0 else "0%")
        
        with col4:
            st.metric("Skipped", skip_count, f"{skip_count / total_tests * 100:.1f}%" if total_tests > 0 else "0%")
        
        # Summary chart
        st.subheader("Results by Test Type")
        chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True)
        
        fig = create_summary_chart(results, chart_type.lower())
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance chart
        st.subheader("Performance by Model")
        fig = create_performance_chart(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent tests table
        st.subheader("Recent Tests")
        
        # Generate summary table
        summary_df = generate_summary_table(results[:10])  # Show only 10 most recent tests
        
        # Apply styling to the table
        def style_status(val):
            if val.lower() in ["pass", "success", "passed", "ok"]:
                return f"background-color: {RESULT_COLORS['pass']}; color: white;"
            elif val.lower() in ["fail", "failure", "failed", "error"]:
                return f"background-color: {RESULT_COLORS['fail']}; color: white;"
            elif val.lower() in ["skip", "skipped", "ignored"]:
                return f"background-color: {RESULT_COLORS['skip']}; color: white;"
            elif val.lower() in ["warning", "warn"]:
                return f"background-color: {RESULT_COLORS['warning']}; color: white;"
            else:
                return f"background-color: {RESULT_COLORS['info']}; color: white;"
        
        styled_df = summary_df.style.applymap(style_status, subset=["Status"])
        
        st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        st.subheader("Model Comparison")
        
        # Filter by test type
        test_types = sorted(set(r.get("test_type", "Unknown") for r in results))
        selected_types = st.multiselect(
            "Filter by Test Type",
            test_types,
            default=test_types
        )
        
        # Filter results
        filtered_results = [r for r in results if r.get("test_type", "Unknown") in selected_types]
        
        if not filtered_results:
            st.warning("No results match the selected filters.")
            return
        
        # Model comparison chart
        fig = create_model_comparison_chart(filtered_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance comparison
        st.subheader("Performance Comparison")
        fig = create_performance_chart(filtered_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        
        # Extract model and test type information
        models = sorted(set(r.get("model", "Unknown") for r in filtered_results))
        test_types = sorted(set(r.get("test_type", "Unknown") for r in filtered_results))
        
        # Create a comparison table
        comparison_data = {
            "Test Type": []
        }
        
        # Add columns for each model
        for model in models:
            comparison_data[model] = []
        
        # Populate the table
        for test_type in test_types:
            comparison_data["Test Type"].append(test_type)
            
            for model in models:
                # Find the most recent result for this model and test type
                model_results = [r for r in filtered_results if r.get("model", "Unknown") == model and r.get("test_type", "Unknown") == test_type]
                model_results.sort(key=lambda x: x.get("timestamp", x.get("file_mtime", 0)), reverse=True)
                
                if model_results:
                    result = model_results[0]
                    status = result.get("status", "Unknown")
                    score = result.get("score", None)
                    
                    if score is not None:
                        value = f"{status} ({score:.2f})"
                    else:
                        value = status
                else:
                    value = "N/A"
                
                comparison_data[model].append(value)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display the table
        st.dataframe(comparison_df, use_container_width=True)
    
    with tab3:
        st.subheader("Test Timeline")
        
        # Timeline chart
        fig = create_timeline_chart(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Test frequency over time
        st.subheader("Test Frequency")
        
        # Extract timestamps
        timestamps = []
        for result in results:
            # Use either timestamp from result or file modification time
            timestamp = result.get("timestamp", result.get("file_mtime", 0))
            
            # Convert timestamp to datetime
            if isinstance(timestamp, (int, float)):
                timestamps.append(datetime.fromtimestamp(timestamp))
            elif isinstance(timestamp, str):
                try:
                    timestamps.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")))
                except ValueError:
                    try:
                        timestamps.append(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"))
                    except ValueError:
                        try:
                            timestamps.append(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
                        except ValueError:
                            timestamps.append(datetime.fromtimestamp(result.get("file_mtime", 0)))
        
        if timestamps:
            # Create DataFrame with timestamps
            df = pd.DataFrame({"Timestamp": timestamps})
            
            # Set timestamp as index
            df.set_index("Timestamp", inplace=True)
            
            # Resample by day and count
            daily_counts = df.resample("D").size()
            
            # Create bar chart
            fig = px.bar(
                daily_counts,
                title="Tests per Day",
                labels={"index": "Date", "value": "Number of Tests"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Test Details")
        
        # Filter results
        col1, col2 = st.columns(2)
        
        with col1:
            filter_model = st.selectbox(
                "Filter by Model",
                ["All"] + sorted(set(r.get("model", "Unknown") for r in results))
            )
        
        with col2:
            filter_type = st.selectbox(
                "Filter by Test Type",
                ["All"] + sorted(set(r.get("test_type", "Unknown") for r in results))
            )
        
        # Apply filters
        filtered_results = results
        
        if filter_model != "All":
            filtered_results = [r for r in filtered_results if r.get("model", "Unknown") == filter_model]
        
        if filter_type != "All":
            filtered_results = [r for r in filtered_results if r.get("test_type", "Unknown") == filter_type]
        
        # Show detailed results
        for i, result in enumerate(filtered_results):
            with st.expander(f"Test #{i+1}: {result.get('test_type', 'Unknown')} - {result.get('model', 'Unknown')}"):
                # Format status with color
                status = result.get("status", "Unknown")
                if status.lower() in ["pass", "success", "passed", "ok"]:
                    status_color = RESULT_COLORS["pass"]
                elif status.lower() in ["fail", "failure", "failed", "error"]:
                    status_color = RESULT_COLORS["fail"]
                elif status.lower() in ["skip", "skipped", "ignored"]:
                    status_color = RESULT_COLORS["skip"]
                elif status.lower() in ["warning", "warn"]:
                    status_color = RESULT_COLORS["warning"]
                else:
                    status_color = RESULT_COLORS["info"]
                
                st.markdown(f"<span style='background-color: {status_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem;'>{status}</span>", unsafe_allow_html=True)
                
                # Result metadata
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Test Type:**", result.get("test_type", "Unknown"))
                    st.write("**Model:**", result.get("model", "Unknown"))
                
                with col2:
                    # Format timestamp
                    timestamp = result.get("timestamp", result.get("file_mtime", 0))
                    if isinstance(timestamp, (int, float)):
                        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    elif isinstance(timestamp, str):
                        timestamp_str = timestamp
                    else:
                        timestamp_str = "Unknown"
                    
                    st.write("**Timestamp:**", timestamp_str)
                    
                    # Format duration
                    duration = result.get("duration", 0)
                    if duration:
                        duration_str = format_duration(duration)
                    else:
                        duration_str = "N/A"
                    
                    st.write("**Duration:**", duration_str)
                
                with col3:
                    score = result.get("score", None)
                    if score is not None:
                        st.write("**Score:**", f"{score:.2f}")
                    
                    st.write("**File:**", result.get("file_name", "Unknown"))
                
                # Test details
                if "prompt" in result:
                    st.subheader("Prompt")
                    st.markdown(f"```\n{result['prompt']}\n```")
                
                if "response" in result:
                    st.subheader("Response")
                    st.markdown(f"```\n{result['response']}\n```")
                
                if "details" in result:
                    st.subheader("Details")
                    if isinstance(result["details"], dict):
                        st.json(result["details"])
                    else:
                        st.write(result["details"])
                
                # Show full result
                with st.expander("Show Raw Data"):
                    st.json(result)

if __name__ == "__main__":
    test_visualization_ui()