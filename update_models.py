# update_models.py
import streamlit as st
from ollama_utils import *

def update_models():
    st.header("🔄 Update Local Ollama Models")
    available_models = get_available_models()
    
    # Create status containers
    status_container = st.empty()
    result_container = st.container()
    
    if st.button("⚡ Update All Models Now!"):
        # Display progress bar
        progress_bar = st.progress(0)
        status_container.info("Starting model updates...")
        
        # Track success and failures
        results = {"updated": [], "skipped": [], "failed": []}
        
        # Calculate total number of models for progress bar
        total_models = len(available_models)
        
        if total_models == 0:
            status_container.warning("No models found to update. Please pull some models first.")
        else:
            for i, model_name in enumerate(available_models):
                # Update progress bar
                progress = (i / total_models)
                progress_bar.progress(progress)
                status_container.info(f"Processing model {i+1} of {total_models}: {model_name}")
                
                # Skip embed models
                if 'embed' in model_name.lower():
                    with result_container:
                        st.write(f"⏩ Skipping embedding model: `{model_name}`")
                    results["skipped"].append(model_name)
                    continue
                    
                # Skip custom models (those with a ':' in the name)
                if ':' in model_name and 'gpt' in model_name:
                    with result_container:
                        st.write(f"⏩ Skipping custom model: `{model_name}`")
                    results["skipped"].append(model_name)
                    continue
                
                # Update the model
                try:
                    with result_container:
                        st.write(f"🔄 Updating model: `{model_name}`")
                    
                    # Get model info before update to compare versions
                    old_info = show_model_info(model_name)
                    old_version = old_info.get("details", {}).get("version", "unknown")
                    
                    # Pull the model
                    pull_result = pull_model(model_name)
                    
                    # Check if model was actually updated
                    new_info = show_model_info(model_name)
                    new_version = new_info.get("details", {}).get("version", "unknown")
                    
                    if old_version != new_version:
                        with result_container:
                            st.success(f"✅ Updated model `{model_name}` from version {old_version} to {new_version}")
                        results["updated"].append(model_name)
                    else:
                        with result_container:
                            st.info(f"✓ Model `{model_name}` is already up to date (version {new_version})")
                        results["updated"].append(model_name)
                except Exception as e:
                    with result_container:
                        st.error(f"❌ Failed to update model `{model_name}`: {str(e)}")
                    results["failed"].append(model_name)
            
            # Complete the progress bar
            progress_bar.progress(1.0)
            
            # Show summary
            status_container.success(f"Update process completed: {len(results['updated'])} updated, {len(results['skipped'])} skipped, {len(results['failed'])} failed")
            
            # If there were failures, show a warning
            if results["failed"]:
                st.warning(f"Some models failed to update: {', '.join(results['failed'])}")
            
            # Show update success message only if all models were updated successfully
            if not results["failed"]:
                st.balloons()
                st.success("All models were updated successfully!")