import streamlit as st
import json
import ollama
import pandas as pd
import time
import logging
import requests
from typing import Dict, Any, List, Optional, Union
from ollama_workbench.providers.ollama_utils import get_available_models

# Try to import json-schema-for-humans, but don't fail if it's not available
try:
    import json_schema_for_humans.generate as jsf
    HAS_JSF = True
except ImportError:
    HAS_JSF = False
    st.warning("json-schema-for-humans package is not installed. Schema visualization will be limited.")
    jsf = None

logger = logging.getLogger(__name__)

# Default JSON schemas for common structured outputs
DEFAULT_SCHEMAS = {
    "person_details": {
        "title": "Person Details",
        "description": "Schema for personal information",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name of the person"
            },
            "age": {
                "type": "integer",
                "description": "Age in years",
                "minimum": 0,
                "maximum": 120
            },
            "email": {
                "type": "string",
                "format": "email",
                "description": "Email address"
            },
            "phone": {
                "type": "string",
                "description": "Phone number"
            },
            "address": {
                "type": "object",
                "description": "Physical address",
                "properties": {
                    "street": {
                        "type": "string",
                        "description": "Street address"
                    },
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "state": {
                        "type": "string",
                        "description": "State or province"
                    },
                    "zip": {
                        "type": "string",
                        "description": "Postal code"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country name"
                    }
                },
                "required": ["street", "city", "state", "zip", "country"]
            }
        },
        "required": ["name", "age", "email"]
    },
    
    "product_info": {
        "title": "Product Information",
        "description": "Schema for product details",
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Unique product identifier"
            },
            "name": {
                "type": "string",
                "description": "Product name"
            },
            "description": {
                "type": "string",
                "description": "Product description"
            },
            "price": {
                "type": "number",
                "description": "Product price",
                "minimum": 0
            },
            "currency": {
                "type": "string",
                "description": "Currency code (e.g., USD, EUR)",
                "enum": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
            },
            "category": {
                "type": "string",
                "description": "Product category"
            },
            "in_stock": {
                "type": "boolean",
                "description": "Whether the product is in stock"
            },
            "attributes": {
                "type": "array",
                "description": "List of product attributes",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Attribute name"
                        },
                        "value": {
                            "type": "string",
                            "description": "Attribute value"
                        }
                    },
                    "required": ["name", "value"]
                }
            }
        },
        "required": ["id", "name", "price", "currency", "in_stock"]
    },
    
    "event": {
        "title": "Event Information",
        "description": "Schema for event details",
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Unique event identifier"
            },
            "title": {
                "type": "string",
                "description": "Event title"
            },
            "description": {
                "type": "string",
                "description": "Event description"
            },
            "start_date": {
                "type": "string",
                "format": "date-time",
                "description": "Event start date and time (ISO 8601 format)"
            },
            "end_date": {
                "type": "string",
                "format": "date-time",
                "description": "Event end date and time (ISO 8601 format)"
            },
            "location": {
                "type": "object",
                "description": "Event location",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Location name"
                    },
                    "address": {
                        "type": "string",
                        "description": "Location address"
                    },
                    "coordinates": {
                        "type": "object",
                        "description": "Geographic coordinates",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "Latitude"
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Longitude"
                            }
                        },
                        "required": ["latitude", "longitude"]
                    }
                },
                "required": ["name", "address"]
            },
            "organizer": {
                "type": "string",
                "description": "Event organizer"
            },
            "attendees": {
                "type": "array",
                "description": "List of attendees",
                "items": {
                    "type": "string",
                    "description": "Attendee name"
                }
            },
            "tags": {
                "type": "array",
                "description": "Event tags or categories",
                "items": {
                    "type": "string",
                    "description": "Tag name"
                }
            }
        },
        "required": ["title", "start_date", "location"]
    },
    
    "article_summary": {
        "title": "Article Summary",
        "description": "Schema for summarizing an article",
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Article title"
            },
            "author": {
                "type": "string",
                "description": "Article author"
            },
            "publication_date": {
                "type": "string",
                "format": "date",
                "description": "Publication date (YYYY-MM-DD)"
            },
            "source": {
                "type": "string",
                "description": "Source or publication name"
            },
            "url": {
                "type": "string",
                "format": "uri",
                "description": "URL of the article"
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of the article"
            },
            "key_points": {
                "type": "array",
                "description": "List of key points from the article",
                "items": {
                    "type": "string",
                    "description": "A key point or takeaway"
                }
            },
            "topics": {
                "type": "array",
                "description": "Main topics covered in the article",
                "items": {
                    "type": "string",
                    "description": "Topic name"
                }
            }
        },
        "required": ["title", "summary", "key_points"]
    },
    
    "data_analysis": {
        "title": "Data Analysis Results",
        "description": "Schema for data analysis output",
        "type": "object",
        "properties": {
            "dataset_name": {
                "type": "string",
                "description": "Name of the dataset analyzed"
            },
            "date_analyzed": {
                "type": "string",
                "format": "date-time",
                "description": "Date and time of analysis (ISO 8601 format)"
            },
            "summary_statistics": {
                "type": "object",
                "description": "Summary statistics of the dataset",
                "properties": {
                    "sample_size": {
                        "type": "integer",
                        "description": "Number of samples or data points"
                    },
                    "variables": {
                        "type": "array",
                        "description": "Variables or features analyzed",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Variable name"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Data type",
                                    "enum": ["numerical", "categorical", "temporal", "other"]
                                },
                                "metrics": {
                                    "type": "object",
                                    "description": "Statistical metrics",
                                    "properties": {
                                        "mean": {
                                            "type": "number",
                                            "description": "Mean value (for numerical variables)"
                                        },
                                        "median": {
                                            "type": "number",
                                            "description": "Median value (for numerical variables)"
                                        },
                                        "std_dev": {
                                            "type": "number",
                                            "description": "Standard deviation (for numerical variables)"
                                        },
                                        "min": {
                                            "type": "number",
                                            "description": "Minimum value (for numerical variables)"
                                        },
                                        "max": {
                                            "type": "number",
                                            "description": "Maximum value (for numerical variables)"
                                        },
                                        "categories": {
                                            "type": "array",
                                            "description": "Categories and their frequencies (for categorical variables)",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "category": {
                                                        "type": "string",
                                                        "description": "Category name"
                                                    },
                                                    "count": {
                                                        "type": "integer",
                                                        "description": "Frequency count"
                                                    },
                                                    "percentage": {
                                                        "type": "number",
                                                        "description": "Percentage of total"
                                                    }
                                                },
                                                "required": ["category", "count"]
                                            }
                                        }
                                    }
                                }
                            },
                            "required": ["name", "type"]
                        }
                    }
                },
                "required": ["sample_size", "variables"]
            },
            "insights": {
                "type": "array",
                "description": "Key insights derived from the analysis",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of the insight"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level (0-1)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "supporting_data": {
                            "type": "string",
                            "description": "Data points or evidence supporting this insight"
                        }
                    },
                    "required": ["description"]
                }
            }
        },
        "required": ["dataset_name", "summary_statistics"]
    },
    
    "recipe": {
        "title": "Recipe Information",
        "description": "Schema for recipe details",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Recipe name"
            },
            "author": {
                "type": "string",
                "description": "Recipe author or creator"
            },
            "description": {
                "type": "string",
                "description": "Brief description of the recipe"
            },
            "prep_time": {
                "type": "integer",
                "description": "Preparation time in minutes"
            },
            "cook_time": {
                "type": "integer",
                "description": "Cooking time in minutes"
            },
            "total_time": {
                "type": "integer",
                "description": "Total time in minutes"
            },
            "servings": {
                "type": "integer",
                "description": "Number of servings"
            },
            "ingredients": {
                "type": "array",
                "description": "List of ingredients",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Ingredient name"
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Ingredient quantity"
                        },
                        "unit": {
                            "type": "string",
                            "description": "Unit of measurement"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Additional notes about the ingredient"
                        }
                    },
                    "required": ["name"]
                }
            },
            "instructions": {
                "type": "array",
                "description": "List of step-by-step instructions",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "integer",
                            "description": "Step number"
                        },
                        "description": {
                            "type": "string",
                            "description": "Step description"
                        },
                        "time": {
                            "type": "integer",
                            "description": "Time for this step in minutes (if applicable)"
                        }
                    },
                    "required": ["step", "description"]
                }
            },
            "categories": {
                "type": "array",
                "description": "Recipe categories (e.g., breakfast, vegetarian)",
                "items": {
                    "type": "string",
                    "description": "Category name"
                }
            },
            "nutrition_facts": {
                "type": "object",
                "description": "Nutritional information per serving",
                "properties": {
                    "calories": {
                        "type": "number",
                        "description": "Calories (kcal)"
                    },
                    "fat": {
                        "type": "number",
                        "description": "Fat (g)"
                    },
                    "carbohydrates": {
                        "type": "number",
                        "description": "Carbohydrates (g)"
                    },
                    "protein": {
                        "type": "number",
                        "description": "Protein (g)"
                    },
                    "fiber": {
                        "type": "number",
                        "description": "Fiber (g)"
                    },
                    "sugar": {
                        "type": "number",
                        "description": "Sugar (g)"
                    },
                    "sodium": {
                        "type": "number",
                        "description": "Sodium (mg)"
                    }
                }
            }
        },
        "required": ["name", "ingredients", "instructions"]
    },
    
    "custom": {
        "title": "Custom Schema",
        "description": "Create your own schema",
        "type": "object",
        "properties": {
            "property1": {
                "type": "string",
                "description": "First property"
            },
            "property2": {
                "type": "number",
                "description": "Second property"
            }
        }
    }
}

def json_editor(schema_data: Dict, key: str) -> Dict:
    """
    Create a JSON editor for modifying a schema.
    
    Args:
        schema_data: The current schema data
        key: A unique key for the editor session
        
    Returns:
        The modified schema data
    """
    # Convert the schema to text for editing
    schema_text = json.dumps(schema_data, indent=2)
    
    # Create a text area for editing
    edited_schema = st.text_area(
        "Edit Schema (JSON)",
        value=schema_text,
        height=300,
        key=f"schema_editor_{key}"
    )
    
    # Validate and parse the edited schema
    try:
        return json.loads(edited_schema)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {str(e)}")
        return schema_data

def visualize_schema(schema: Dict) -> None:
    """
    Visualize a JSON schema in a user-friendly way.
    
    Args:
        schema: The JSON schema to visualize
    """
    # Generate HTML for the schema
    if HAS_JSF:
        try:
            # Use json-schema-for-humans library to generate a nice visualization
            html = jsf.generate_from_schema(schema, "md")
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            # Fallback to basic visualization
            st.json(schema)
            st.warning(f"Could not generate enhanced visualization: {str(e)}")
    else:
        # Basic fallback if json-schema-for-humans is not available
        st.json(schema)
        st.info("Install json-schema-for-humans for enhanced schema visualization")

def generate_json_from_text(text: str, schema: Dict, model: str, temperature: float = 0.7) -> Dict:
    """
    Generate structured JSON from text using a model and schema.
    
    Args:
        text: The input text to structure
        schema: The JSON schema to follow
        model: The model to use for generation
        temperature: The temperature parameter for generation
        
    Returns:
        The generated JSON object
    """
    try:
        # Prepare the prompt
        prompt = f"""
Your task is to create a JSON object based on the following text. 
Follow the provided JSON Schema exactly, including all required fields.
Ensure the output is valid JSON.

TEXT:
{text}

JSON SCHEMA:
{json.dumps(schema, indent=2)}

Respond with ONLY the JSON object, no other text. The JSON should be valid and follow the schema exactly.
"""
        
        # Get client from ollama_utils
        from ollama_workbench.providers.ollama_utils import call_ollama_endpoint, get_ollama_client
        
        if not model:
            raise ValueError("No model selected. Please select a valid model.")
        
        # Log what we're doing
        logger.info(f"Generating JSON from text using model: {model}")
        
        # Try multiple methods to interact with Ollama
        try:
            # First try direct command line
            try:
                import subprocess
                logger.info(f"Trying CLI method for model {model}")
                result = subprocess.run(['ollama', 'run', model, prompt], capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    response_text = result.stdout.strip()
                    logger.info(f"Successfully used CLI for model {model}")
                else:
                    # CLI failed, try API methods
                    raise Exception(f"CLI error: {result.stderr}")
            except Exception as cli_error:
                # CLI method failed, try API methods
                logger.warning(f"CLI method failed: {cli_error}, trying API methods")

                client = get_ollama_client()
                if client:
                    # New Client API
                    logger.info(f"Using Client API for model {model}")
                    response = client.generate(
                        model=model,
                        prompt=prompt,
                        options={
                            "temperature": temperature,
                            "num_predict": 2048
                        }
                    )
                    if not response or "response" not in response:
                        logger.warning(f"Unexpected response format: {response}")
                        raise ValueError("Received an invalid response from the model.")
                    response_text = response.get("response", "")
                    logger.info("Client API request completed successfully")
                else:
                    # Client unavailable, fall back to version-independent function
                    logger.info("Client not available, using call_ollama_endpoint")
                    response_text, _, _, _, _ = call_ollama_endpoint(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=2048
                    )
                    logger.info("Successfully received response using call_ollama_endpoint")
        except Exception as api_error:
            logger.warning(f"Error using API methods: {api_error}, trying final CLI fallback")
            # Final fallback - try CLI again with simpler approach
            try:
                import subprocess
                cmd = ['ollama', 'run', model, prompt]
                logger.info(f"Running final CLI fallback command: {cmd}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    response_text = result.stdout.strip()
                    logger.info(f"Final CLI fallback successful for model {model}")
                else:
                    raise Exception(f"CLI error: {result.stderr}")
            except Exception as final_error:
                logger.error(f"All methods failed: {final_error}")
                raise ValueError(f"Failed to generate JSON with model {model}: {str(final_error)}")
        
        # Try to find JSON in the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")
        
        if json_start != -1 and json_end != -1:
            json_text = response_text[json_start:json_end + 1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                # If parsing fails, try to fix common issues
                json_text = json_text.replace("'", '"')  # Replace single quotes
                try:
                    return json.loads(json_text)
                except Exception:
                    st.error("Failed to parse JSON from model response")
                    logger.error(f"Failed to parse JSON: {response_text}")
                    return {}
        else:
            st.error("No JSON found in model response")
            logger.error(f"No JSON found in: {response_text}")
            return {}
    
    except Exception as e:
        st.error(f"Error generating JSON: {str(e)}")
        logger.error(f"Error generating JSON: {str(e)}")
        return {}

def structured_output_ui():
    """Main UI for structured output generation"""
    st.title("🔍 Structured Output Generator")
    st.write("Generate structured JSON data from text using a schema")
    
    # Get available models
    try:
        # Get models directly from the command line as a fallback
        import subprocess
        
        try:
            # First try the API approach
            available_models = get_available_models()
            logging.info(f"Got {len(available_models)} models from API")
            
            # If API returns empty but Ollama is running, try CLI fallback
            if not available_models:
                try:
                    # Run ollama list command to get models
                    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header line
                    available_models = []
                    for line in lines:
                        if line.strip():
                            # The model name is the first column, which may contain colons and other special chars
                            parts = line.split()
                            if parts:  # Make sure line has content
                                model_name = parts[0]
                                available_models.append(model_name)
                    logging.info(f"Got {len(available_models)} models from CLI fallback")
                except subprocess.CalledProcessError as cli_error:
                    logging.error(f"CLI fallback failed: {cli_error}")
                    # Continue with empty list to show error below
                except Exception as cli_ex:
                    logging.error(f"Error in CLI fallback: {cli_ex}")
                    # Continue with empty list
            
            # Check if there are any models available after all attempts
            if not available_models:
                st.error("No Ollama models found. Please pull some models first.")
                st.info("You can pull models using the 'Pull Models' tab in the sidebar or by running `ollama pull <model_name>` in your terminal.")
                return
                
        except Exception as api_error:
            logging.error(f"API error fetching models: {api_error}")
            
            # Try CLI approach as fallback
            try:
                # Run ollama list command to get models
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
                lines = result.stdout.strip().split('\n')[1:]  # Skip header line
                available_models = []
                for line in lines:
                    if line.strip():
                        # The model name is the first column, which may contain colons and other special chars
                        parts = line.split()
                        if parts:  # Make sure line has content
                            model_name = parts[0]
                            available_models.append(model_name)
                logging.info(f"Got {len(available_models)} models from CLI direct")
                
                if not available_models:
                    st.error("No Ollama models found. Please pull some models first.")
                    st.info("You can pull models using the 'Pull Models' tab in the sidebar or by running `ollama pull <model_name>` in your terminal.")
                    return
            except subprocess.CalledProcessError as cli_error:
                st.error(f"Failed to get models via CLI: {cli_error}")
                st.info("Please make sure Ollama is installed and running.")
                return
            except Exception as cli_ex:
                st.error(f"Error fetching models: {cli_ex}")
                st.info("Please make sure Ollama is installed and running.")
                return
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        st.info("This could be due to the Ollama server not running or an issue with the Ollama API. Please make sure Ollama is running and try again.")
        available_models = []
        return
    
    # Initialize session state
    if "structured_output_schema" not in st.session_state:
        st.session_state.structured_output_schema = DEFAULT_SCHEMAS["person_details"]
    if "structured_output_result" not in st.session_state:
        st.session_state.structured_output_result = {}
    if "structured_output_history" not in st.session_state:
        st.session_state.structured_output_history = []
    
    # Layout with two columns
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Schema selection and editing
        st.subheader("1. Select Schema")
        
        selected_schema = st.selectbox(
            "Choose a schema template",
            list(DEFAULT_SCHEMAS.keys()),
            key="schema_selector",
            format_func=lambda x: DEFAULT_SCHEMAS[x]["title"]
        )
        
        # Load the selected schema
        current_schema = DEFAULT_SCHEMAS[selected_schema].copy()
        
        # Schema editor
        st.subheader("2. Edit Schema (Optional)")
        edited_schema = json_editor(current_schema, "main_schema")
        
        # Update the session state with the edited schema
        st.session_state.structured_output_schema = edited_schema
        
        # Schema visualization
        with st.expander("Schema Visualization", expanded=False):
            st.subheader("Schema Structure")
            visualize_schema(edited_schema)
        
        # Save custom schema
        if selected_schema != "custom" and st.button("Save as Custom Schema"):
            DEFAULT_SCHEMAS["custom"] = edited_schema.copy()
            st.success("Schema saved to Custom Schema template")
        
        # Model selection and parameters
        st.subheader("3. Model Settings")
        
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            key="model_selector"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature_slider"
        )
        
        # Input text
        st.subheader("4. Input Text")
        input_text = st.text_area(
            "Text to structure",
            height=200,
            help="Enter the text that should be structured according to the schema",
            key="input_text"
        )
        
        # Generate button
        generate_button = st.button("Generate Structured Output", key="generate_button")
    
    with col2:
        # Output area
        st.subheader("Structured Output")
        
        # Generate output if button is clicked
        if generate_button and input_text:
            if not selected_model:
                st.error("Please select a model before generating structured output.")
            else:
                try:
                    with st.spinner("Generating structured output..."):
                        try:
                            # Log the attempt
                            logger.info(f"Attempting to generate structured output with model: {selected_model}")
                            
                            result = generate_json_from_text(
                                text=input_text,
                                schema=st.session_state.structured_output_schema,
                                model=selected_model,
                                temperature=temperature
                            )
                            
                            # Store the result
                            st.session_state.structured_output_result = result
                            
                            # Add to history
                            if result:
                                history_item = {
                                    "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                                    "schema": edited_schema.get("title", "Untitled Schema"),
                                    "result": result,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                st.session_state.structured_output_history.append(history_item)
                                logger.info("Successfully generated structured output and added to history")
                        except Exception as gen_error:
                            st.error(f"Error generating structured output: {str(gen_error)}")
                            logger.error(f"Error generating structured output: {gen_error}", exc_info=True)
                except Exception as outer_error:
                    st.error(f"An unexpected error occurred: {str(outer_error)}")
                    logger.error(f"Outer error in structured output generation: {outer_error}", exc_info=True)
        
        # Display the result
        if st.session_state.structured_output_result:
            # Format options
            output_format = st.radio(
                "Output Format",
                ["JSON", "Table", "Both"],
                horizontal=True,
                key="output_format"
            )
            
            # Show JSON output
            if output_format in ["JSON", "Both"]:
                st.json(st.session_state.structured_output_result)
            
            # Show table output for flat structures or first level
            if output_format in ["Table", "Both"]:
                try:
                    # Create a flattened version for display
                    flat_result = {}
                    
                    def flatten_dict(d, parent_key=""):
                        for k, v in d.items():
                            new_key = f"{parent_key}.{k}" if parent_key else k
                            if isinstance(v, dict) and len(v) > 0:
                                # For objects, only go one level deep
                                if not parent_key:
                                    flatten_dict(v, new_key)
                                else:
                                    flat_result[new_key] = str(v)
                            elif isinstance(v, list):
                                # For lists, show count and sample
                                if len(v) > 0:
                                    flat_result[new_key] = f"[{len(v)} items] e.g. {v[0]}"
                                else:
                                    flat_result[new_key] = "[]"
                            else:
                                flat_result[new_key] = v
                    
                    flatten_dict(st.session_state.structured_output_result)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(list(flat_result.items()), columns=["Field", "Value"])
                    st.table(df)
                except Exception as e:
                    st.error(f"Could not display as table: {str(e)}")
            
            # Copy to clipboard button
            if st.button("Copy JSON to Clipboard", key="copy_button"):
                try:
                    json_str = json.dumps(st.session_state.structured_output_result, indent=2)
                    st.write("JSON copied to clipboard!")
                    st.code(json_str)
                except Exception as e:
                    st.error(f"Error copying to clipboard: {str(e)}")
        else:
            st.info("Generate structured output to see results here")
        
        # History
        if st.session_state.structured_output_history:
            with st.expander("Generation History", expanded=False):
                for i, item in enumerate(reversed(st.session_state.structured_output_history)):
                    st.markdown(f"**{item['timestamp']}**: {item['input']}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(f"View Result", key=f"view_{i}"):
                            st.session_state.structured_output_result = item["result"]
                            st.rerun()
                    
                    st.markdown("---")

# For direct execution
if __name__ == "__main__":
    structured_output_ui()