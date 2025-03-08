"""
Global VRM Loader module for managing VRM model loading functionality.
This module provides a singleton instance for loading and managing VRM models.
"""

class VRMLoader:
    def __init__(self):
        self._loaded_models = {}

    def load_model(self, key, vrm_model_path):
        """
        Load a VRM model from the specified path and store it with the given key.
        
        Args:
            key (str): The identifier key for the model
            vrm_model_path (str): Path to the VRM model file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Here you would implement the actual VRM loading logic
            # For now, we'll just store the path as a placeholder
            self._loaded_models[key] = vrm_model_path
            return True
        except Exception as e:
            print(f"Error loading VRM model: {e}")
            return False

    def get_model(self, key):
        """
        Retrieve a loaded VRM model by its key.
        
        Args:
            key (str): The identifier key for the model
            
        Returns:
            The loaded model or None if not found
        """
        return self._loaded_models.get(key)

    def unload_model(self, key):
        """
        Unload a VRM model from memory.
        
        Args:
            key (str): The identifier key for the model
            
        Returns:
            bool: True if unloading was successful, False if model wasn't loaded
        """
        if key in self._loaded_models:
            del self._loaded_models[key]
            return True
        return False

# Create a global instance
global_vrm_loader = VRMLoader()
