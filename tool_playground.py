import streamlit as st
import json
import ollama
import time
import pandas as pd
import base64
import io
import logging
import random
import uuid
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union
from ollama_utils import get_available_models
from mcp_tools import get_available_mcp_tools, execute_mcp_tool
from model_capability_registry import filter_models_by_capability, is_tools_capable

# Setup logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()


# Helper function to ensure objects are JSON serializable
def sanitize_for_json(obj):
    # Helper to convert Python objects to JSON serializable types
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if k != "_client"}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, 'to_dict'):  # Handle objects with to_dict method
        return sanitize_for_json(obj.to_dict())
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return sanitize_for_json({k: v for k, v in obj.__dict__.items() 
                                if not k.startswith('_')})
    # Special handling for ToolCall objects (Ollama client objects)
    elif hasattr(obj, 'function') and hasattr(obj, 'id'):
        # Create a dict representation of the ToolCall
        tool_dict = {'id': obj.id if hasattr(obj, 'id') else str(uuid.uuid4())}
        if hasattr(obj, 'function'):
            function_dict = {}
            if hasattr(obj.function, 'name'):
                function_dict['name'] = obj.function.name
            if hasattr(obj.function, 'arguments'):
                function_dict['arguments'] = obj.function.arguments
            tool_dict['function'] = function_dict
        return tool_dict
    else:
        try:
            # Test if it's JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            # Convert to string if not serializable
            return str(obj)
# Predefined tool templates with example implementations
TOOL_TEMPLATES = {
    "json": {
        "type": "function",
        "function": {
            "name": "json_tool",
            "description": "Parse, validate, and format JSON data. Can also repair malformed JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The JSON text to parse, validate, or repair"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["parse", "validate", "format", "repair"],
                        "description": "The operation to perform on the JSON text"
                    },
                    "indent": {
                        "type": "integer",
                        "description": "Number of spaces for indentation when formatting"
                    }
                },
                "required": ["text"]
            }
        }
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic calculations on a given expression or basic operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (preferred), e.g. '2+2*3'"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform (alternative to expression)"
                    },
                    "a": {
                        "type": "number",
                        "description": "The first number (use with operation)"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number (use with operation)"
                    }
                },
                "required": []
            }
        }
    },
    "weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }
        }
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on the internet",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of search results to return"
                    }
                },
                "required": ["query"]
            }
        }
    },
    "plot": {
        "type": "function",
        "function": {
            "name": "create_plot",
            "description": "Create a simple plot from data",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "pie"],
                        "description": "The type of plot to create"
                    },
                    "x_data": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "The x-axis data points"
                    },
                    "y_data": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "The y-axis data points"
                    },
                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Labels for the data points (required for pie charts)"
                    },
                    "title": {
                        "type": "string",
                        "description": "The title of the plot"
                    }
                },
                "required": ["plot_type"]
            }
        }
    },
    "database": {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query a fictional database",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": ["users", "products", "orders"],
                        "description": "The table to query"
                    },
                    "filter": {
                        "type": "object",
                        "description": "Filter criteria for the query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["table"]
            }
        }
    }
}

# Tool implementation functions
def calculator_impl(operation: str = None, a: float = None, b: float = None, expression: str = None) -> str:
    """Implementation of the calculator tool with enhanced natural language processing
    
    Supports multiple input modes:
    1. Basic operations with a and b: add, subtract, multiply, divide
    2. Expression evaluation: e.g. "2+2*3"
    3. Natural language queries: e.g. "what is 342 multiplied by 15"
    """
    # Log inputs for debugging
    logger.info(f"Calculator called with: operation={operation}, a={a}, b={b}, expression={expression}")
    
    # PHASE 1: Try to extract information from natural language if needed
    # This helps when LLMs don't follow the schema strictly
    
    # If we have a natural language query in operation but no a, b or expression
    if isinstance(operation, str) and not all([a, b, expression]):
        import re
        operation_lower = operation.lower()
        
        # STEP 1: Check for multiplication patterns
        if "multipl" in operation_lower or " x " in operation_lower or "times" in operation_lower:
            # Common multiplication patterns
            mult_patterns = [
                r'(\d+)\s*(?:multiplied by|times|x)\s*(\d+)',  # "342 multiplied by 15"
                r'(?:multiply|calculate|compute|what\s+is)\s+(\d+)\s*(?:and|by|with|times|\*|x)\s*(\d+)',  # "multiply 342 by 15"
            ]
            
            for pattern in mult_patterns:
                match = re.search(pattern, operation_lower)
                if match:
                    a = float(match.group(1))
                    b = float(match.group(2))
                    operation = "multiply"
                    logger.info(f"Extracted multiplication: {a} * {b}")
                    break
            
        # STEP 2: Check for other arithmetic operations
        if not a or not b:
            # Addition
            if "add" in operation_lower or "sum" in operation_lower or "plus" in operation_lower:
                add_match = re.search(r'(\d+)\s*(?:added to|plus|\+)\s*(\d+)', operation_lower)
                if add_match:
                    a = float(add_match.group(1))
                    b = float(add_match.group(2))
                    operation = "add"
                    logger.info(f"Extracted addition: {a} + {b}")
            
            # Subtraction
            elif "subtract" in operation_lower or "minus" in operation_lower:
                sub_match = re.search(r'(\d+)\s*(?:minus|subtracted by|-)\s*(\d+)', operation_lower)
                if sub_match:
                    a = float(sub_match.group(1))
                    b = float(sub_match.group(2))
                    operation = "subtract"
                    logger.info(f"Extracted subtraction: {a} - {b}")
            
            # Division
            elif "divide" in operation_lower:
                div_match = re.search(r'(\d+)\s*(?:divided by|/)\s*(\d+)', operation_lower)
                if div_match:
                    a = float(div_match.group(1))
                    b = float(div_match.group(2))
                    operation = "divide"
                    logger.info(f"Extracted division: {a} / {b}")
        
        # STEP 3: Try to extract general math expressions
        if not a or not b or not expression:
            # Look for patterns like "what is 2 + 3" or "calculate 5 * 10"
            expr_patterns = [
                r'(?:what\s*(?:is|\'s)|calculate|compute|solve)\s*([\d\s\+\-\*\/\(\)\.]+)',  # "what is 2 + 3"
                r'([\d\s\+\-\*\/\(\)\.]+)\s*(?:=|equals)',  # "2 + 3 ="
            ]
            
            for pattern in expr_patterns:
                expr_match = re.search(pattern, operation_lower)
                if expr_match:
                    expr_text = expr_match.group(1).strip()
                    # Clean up the expression - remove spaces
                    expression = re.sub(r'\s+', '', expr_text)
                    logger.info(f"Extracted expression: {expression}")
                    break
        
        # STEP 4: If we still don't have values but have numbers in the text, use them
        if (not a or not b) and not expression:
            numbers = re.findall(r'\d+', operation)
            if len(numbers) >= 2:
                a = float(numbers[0]) 
                b = float(numbers[1])
                
                # Try to determine operation from context
                if "multi" in operation_lower or "product" in operation_lower or "*" in operation or "x" in operation_lower:
                    operation = "multiply"
                elif "add" in operation_lower or "sum" in operation_lower or "+" in operation:
                    operation = "add"
                elif "subtract" in operation_lower or "diff" in operation_lower or "-" in operation:
                    operation = "subtract"
                elif "div" in operation_lower or "/" in operation:
                    operation = "divide"
                else:
                    # Default to multiplication for queries like "what is 342 and 15"
                    operation = "multiply"
                    
                logger.info(f"Extracted numbers and guessed operation: {operation}({a}, {b})")
        
        # STEP 5: If we have numbers but no explicit operation, check for direct formula
        if not expression and not operation and not (a and b):
            # Look for direct formula like "342 * 15"
            formula_match = re.search(r'(\d+)\s*[\+\-\*\/x]\s*(\d+)', operation_lower)
            if formula_match:
                formula = formula_match.group(0)
                # Replace 'x' with '*' for evaluation
                formula = formula.replace('x', '*')
                expression = formula
                logger.info(f"Extracted formula: {expression}")
    
    # PHASE 2: Evaluate the calculation using the information we've gathered
    
    # MODE 1: Expression evaluation (preferred)
    if expression is not None:
        try:
            # WARNING: using eval can be dangerous in production; use a safe math parser instead
            # Replace 'x' with '*' for multiplication
            expression_safe = expression.replace('x', '*').replace('X', '*')
            result = eval(expression_safe)
            return f"Result: {result}"
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            # Don't return error yet, try other methods
    
    # MODE 2: Basic operations with explicit a, b, and operation
    if operation is not None and a is not None and b is not None:
        try:
            if operation == "add" or operation == "+":
                return f"Result: {a + b}"
            elif operation == "subtract" or operation == "-":
                return f"Result: {a - b}"
            elif operation == "multiply" or operation == "*" or operation == "×" or operation == "x":
                return f"Result: {a * b}"
            elif operation == "divide" or operation == "/":
                if b == 0:
                    return "Error: Division by zero"
                return f"Result: {a / b}"
            else:
                logger.warning(f"Unknown operation '{operation}', trying to guess")
                # Try to guess based on keywords
                if "multi" in str(operation).lower():
                    return f"Result: {a * b}"
                elif "add" in str(operation).lower() or "sum" in str(operation).lower():
                    return f"Result: {a + b}"
                elif "sub" in str(operation).lower() or "minus" in str(operation).lower():
                    return f"Result: {a - b}"
                elif "div" in str(operation).lower():
                    if b == 0:
                        return "Error: Division by zero"
                    return f"Result: {a / b}"
                else:
                    logger.error(f"Could not determine operation type from '{operation}'")
        except Exception as e:
            logger.error(f"Error performing operation {operation}({a}, {b}): {e}")
            # Don't return error yet, try other methods
    
    # MODE 3: Evaluate operation string directly if it looks like a formula
    if operation is not None and isinstance(operation, str):
        try:
            # Check if the operation string contains basic math operators
            if any(char in operation for char in "+-*/x") and re.search(r'\d', operation):
                # This looks like an expression, try to evaluate it
                logger.info(f"Attempting to evaluate operation as expression: {operation}")
                # Replace 'x' with '*' for multiplication
                operation_safe = operation.replace('x', '*').replace('X', '*')
                result = eval(operation_safe)
                return f"Result: {result}"
        except Exception as e:
            logger.error(f"Failed to evaluate as expression: {e}")
    
    # MODE 4: Last resort - try to parse the operation text for "X multiplied by Y"
    if isinstance(operation, str):
        try:
            import re
            # Look specifically for "X multiplied by Y" pattern
            mult_match = re.search(r'(\d+)\s*(?:multiplied by|times|x)\s*(\d+)', operation, re.IGNORECASE)
            if mult_match:
                num1 = float(mult_match.group(1))
                num2 = float(mult_match.group(2))
                result = num1 * num2
                return f"Result: {result}"
            
            # Check for "What is X * Y" pattern
            general_match = re.search(r'(?:what is|calculate|compute)?\s*(\d+)\s*([\+\-\*x\/])\s*(\d+)', operation, re.IGNORECASE)
            if general_match:
                num1 = float(general_match.group(1))
                operator = general_match.group(2)
                num2 = float(general_match.group(3))
                
                if operator == '+':
                    result = num1 + num2
                elif operator == '-':
                    result = num1 - num2
                elif operator in ['*', 'x', 'X']:
                    result = num1 * num2
                elif operator == '/':
                    if num2 == 0:
                        return "Error: Division by zero"
                    result = num1 / num2
                    
                return f"Result: {result}"
        except Exception as e:
            logger.error(f"Failed final attempt to parse operation text: {e}")
    
    # If we get here, we've failed to extract a computable expression from the inputs
    return "Error: Missing required parameters. Either provide 'expression' or all of 'operation', 'a', and 'b'. You can also use natural language like '342 multiplied by 15'."

def weather_impl(location: str = None, unit: str = "celsius") -> str:
    """Implementation of the weather tool with mock data
    
    Also handles inputs in the format "weather in Pittsburgh" or just "Pittsburgh"
    """
    # Extract location from natural language if needed (e.g., "weather in Pittsburgh")
    if location is None or location.strip() == "":
        return "Error: Please provide a location to get weather information."
        
    # Handle formats like "weather in Pittsburgh" or "what's the weather in Pittsburgh"
    if "weather in " in location.lower() or "weather for " in location.lower():
        import re
        match = re.search(r'weather (?:in|for) (.+)', location.lower())
        if match:
            location = match.group(1).strip()
    
    # Clean up location name (remove punctuation, capitalize first letter)
    location = location.strip()
    if location.endswith(".") or location.endswith("?") or location.endswith("!"):
        location = location[:-1]
    location = location.capitalize()
    
    # Generate mock weather data
    temp = random.uniform(-10, 40)
    if unit == "fahrenheit":
        temp = temp * 9/5 + 32
    
    conditions = random.choice([
        "Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorms", 
        "Snowing", "Foggy", "Windy", "Clear"
    ])
    
    humidity = random.randint(30, 95)
    wind_speed = random.uniform(0, 30)
    
    return f"""Weather for {location}:
Temperature: {temp:.1f}° {'F' if unit == 'fahrenheit' else 'C'}
Conditions: {conditions}
Humidity: {humidity}%
Wind Speed: {wind_speed:.1f} km/h

Forecast: Similar conditions expected for the next 24 hours.
This is simulated weather data for demonstration purposes."""

def search_impl(query: str, num_results: int = 3) -> str:
    """Implementation of the search tool with mock results"""
    # List of mock search results
    mock_results = [
        {"title": f"Result for '{query}' - Article 1", "snippet": f"This is a sample search result about {query} with some relevant information that might be useful.", "url": f"https://example.com/article-1-about-{query.replace(' ', '-')}"},
        {"title": f"Understanding {query} - Complete Guide", "snippet": f"A comprehensive guide to understanding {query} with examples, case studies, and expert opinions.", "url": f"https://example.com/guide-to-{query.replace(' ', '-')}"},
        {"title": f"{query} Explained - For Beginners", "snippet": f"An explanation of {query} targeted at beginners, with simple language and helpful diagrams.", "url": f"https://example.com/beginners-{query.replace(' ', '-')}"},
        {"title": f"Advanced Topics in {query}", "snippet": f"Exploring advanced topics related to {query} for experts and professionals in the field.", "url": f"https://example.com/advanced-{query.replace(' ', '-')}"},
        {"title": f"The History of {query}", "snippet": f"A historical overview of how {query} developed over time and its major milestones.", "url": f"https://example.com/history-of-{query.replace(' ', '-')}"},
        {"title": f"{query} vs. Alternatives - Comparison", "snippet": f"A detailed comparison between {query} and alternative approaches or solutions.", "url": f"https://example.com/comparing-{query.replace(' ', '-')}"},
        {"title": f"Latest Research on {query}", "snippet": f"Recent studies and research findings related to {query} from leading institutions.", "url": f"https://example.com/research-{query.replace(' ', '-')}"},
        {"title": f"How to Implement {query} - Tutorial", "snippet": f"Step-by-step tutorial on implementing or using {query} in practical scenarios.", "url": f"https://example.com/tutorial-{query.replace(' ', '-')}"},
        {"title": f"Common Mistakes When Working with {query}", "snippet": f"A list of common errors and misconceptions about {query} and how to avoid them.", "url": f"https://example.com/mistakes-{query.replace(' ', '-')}"},
        {"title": f"The Future of {query}", "snippet": f"Predictions and trends about how {query} will evolve in the coming years.", "url": f"https://example.com/future-of-{query.replace(' ', '-')}"},
    ]
    
    # Shuffle and limit results
    random.shuffle(mock_results)
    results = mock_results[:min(num_results, len(mock_results))]
    
    # Format the results
    formatted_results = f"Search results for: {query}\n\n"
    for i, result in enumerate(results, 1):
        formatted_results += f"{i}. {result['title']}\n"
        formatted_results += f"   {result['snippet']}\n"
        formatted_results += f"   URL: {result['url']}\n\n"
    
    return formatted_results.strip()

def create_plot_impl(plot_type: str, x_data: Optional[List[float]] = None, 
                    y_data: Optional[List[float]] = None, labels: Optional[List[str]] = None,
                    title: str = "Generated Plot") -> str:
    """Implementation of the plot creation tool"""
    # Generate mock data if not provided
    if plot_type in ["line", "bar", "scatter"] and (x_data is None or y_data is None):
        x_data = list(range(1, 6))
        y_data = [random.uniform(1, 10) for _ in range(5)]
    
    if plot_type == "pie" and (labels is None):
        labels = ["Category A", "Category B", "Category C", "Category D", "Category E"]
        y_data = [random.uniform(5, 20) for _ in range(len(labels))]
    
    # Create a plot based on the type
    plt.figure(figsize=(10, 6))
    
    if plot_type == "bar":
        plt.bar(x_data, y_data)
        plt.xlabel("X Values")
        plt.ylabel("Y Values")
    elif plot_type == "line":
        plt.plot(x_data, y_data)
        plt.xlabel("X Values")
        plt.ylabel("Y Values")
    elif plot_type == "scatter":
        plt.scatter(x_data, y_data)
        plt.xlabel("X Values")
        plt.ylabel("Y Values")
    elif plot_type == "pie":
        plt.pie(y_data, labels=labels, autopct='%1.1f%%')
    
    plt.title(title)
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the plot as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Close the plot to free memory
    plt.close()
    
    return f"""Plot created successfully:

![{title}](data:image/png;base64,{img_str})

Plot type: {plot_type}
Title: {title}
"""

def query_database_impl(table: str, filter: Optional[Dict[str, Any]] = None, limit: int = 5) -> str:
    """Implementation of the database query tool with mock data"""
    # Mock database tables
    mock_db = {
        "users": [
            {"id": 1, "name": "Alice Smith", "email": "alice@example.com", "age": 28, "city": "New York"},
            {"id": 2, "name": "Bob Johnson", "email": "bob@example.com", "age": 35, "city": "Los Angeles"},
            {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "age": 42, "city": "Chicago"},
            {"id": 4, "name": "Diana Miller", "email": "diana@example.com", "age": 31, "city": "Miami"},
            {"id": 5, "name": "Edward Davis", "email": "edward@example.com", "age": 24, "city": "Boston"},
            {"id": 6, "name": "Fiona Wilson", "email": "fiona@example.com", "age": 29, "city": "Seattle"},
            {"id": 7, "name": "George Martin", "email": "george@example.com", "age": 38, "city": "Denver"},
            {"id": 8, "name": "Hannah Clark", "email": "hannah@example.com", "age": 27, "city": "Portland"},
            {"id": 9, "name": "Ian Taylor", "email": "ian@example.com", "age": 33, "city": "Austin"},
            {"id": 10, "name": "Julia Adams", "email": "julia@example.com", "age": 26, "city": "San Francisco"}
        ],
        "products": [
            {"id": 1, "name": "Laptop Pro", "category": "Electronics", "price": 1299.99, "stock": 45},
            {"id": 2, "name": "Smartphone X", "category": "Electronics", "price": 799.99, "stock": 120},
            {"id": 3, "name": "Coffee Maker Deluxe", "category": "Kitchen", "price": 89.95, "stock": 30},
            {"id": 4, "name": "Designer Desk Chair", "category": "Furniture", "price": 249.50, "stock": 15},
            {"id": 5, "name": "Wireless Earbuds", "category": "Electronics", "price": 129.99, "stock": 200},
            {"id": 6, "name": "Blender Pro", "category": "Kitchen", "price": 79.99, "stock": 25},
            {"id": 7, "name": "LED TV 55\"", "category": "Electronics", "price": 549.99, "stock": 18},
            {"id": 8, "name": "Ergonomic Keyboard", "category": "Electronics", "price": 119.99, "stock": 75},
            {"id": 9, "name": "Stainless Steel Cookware Set", "category": "Kitchen", "price": 199.99, "stock": 12},
            {"id": 10, "name": "Bookshelf", "category": "Furniture", "price": 179.99, "stock": 8}
        ],
        "orders": [
            {"id": 1, "user_id": 3, "product_id": 7, "quantity": 1, "total": 549.99, "date": "2023-11-05"},
            {"id": 2, "user_id": 1, "product_id": 2, "quantity": 1, "total": 799.99, "date": "2023-11-04"},
            {"id": 3, "user_id": 5, "product_id": 3, "quantity": 2, "total": 179.90, "date": "2023-11-03"},
            {"id": 4, "user_id": 2, "product_id": 5, "quantity": 1, "total": 129.99, "date": "2023-11-02"},
            {"id": 5, "user_id": 8, "product_id": 1, "quantity": 1, "total": 1299.99, "date": "2023-11-01"},
            {"id": 6, "user_id": 4, "product_id": 9, "quantity": 1, "total": 199.99, "date": "2023-10-31"},
            {"id": 7, "user_id": 7, "product_id": 8, "quantity": 2, "total": 239.98, "date": "2023-10-30"},
            {"id": 8, "user_id": 9, "product_id": 4, "quantity": 1, "total": 249.50, "date": "2023-10-29"},
            {"id": 9, "user_id": 6, "product_id": 10, "quantity": 1, "total": 179.99, "date": "2023-10-28"},
            {"id": 10, "user_id": 10, "product_id": 6, "quantity": 1, "total": 79.99, "date": "2023-10-27"}
        ]
    }
    
    if table not in mock_db:
        return f"Error: Table '{table}' does not exist. Available tables: {', '.join(mock_db.keys())}"
    
    # Filter the results
    results = mock_db[table]
    if filter:
        filtered_results = []
        for row in results:
            match = True
            for key, value in filter.items():
                if key not in row or row[key] != value:
                    match = False
                    break
            if match:
                filtered_results.append(row)
        results = filtered_results
    
    # Limit the results
    results = results[:limit]
    
    # Format the results as a table
    if len(results) == 0:
        return f"No results found in table '{table}' with the specified filter."
    
    # Get all column names from the first result
    columns = list(results[0].keys())
    
    # Create a DataFrame for nicer formatting
    df = pd.DataFrame(results)
    
    return f"""Query results from table '{table}':

{df.to_markdown(index=False)}

{len(results)} row(s) returned from a total of {len(mock_db[table])} rows in table.
This is simulated database data for demonstration purposes."""

# JSON tool implementation function
def json_tool_impl(text: str, operation: str = "parse", indent: int = 2) -> str:
    """Implementation of the JSON tool with robust error handling
    
    This tool can parse, validate, format, and repair malformed JSON data.
    It includes multiple recovery strategies for common JSON formatting errors.
    """
    logger.info(f"JSON tool called with: text={text[:100]}{'...' if len(text) > 100 else ''}, operation={operation}, indent={indent}")
    
    if not text or text.strip() == "":
        return "Error: No JSON text provided."
    
    # Initialize response components
    result = ""
    errors = []
    warnings = []
    fixes_applied = []
    
    # Function to attempt JSON parsing with error recovery
    def parse_json_with_recovery(json_text):
        # First try parsing as-is
        try:
            parsed = json.loads(json_text)
            return {"success": True, "data": parsed}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            errors.append(f"Initial parse error: {str(e)}")
            
            # Recovery attempt 1: Fix missing quotes around property names
            try:
                import re
                fixed_text = re.sub(r'(\s*?)(\w+)(\s*?):(\s*?)', r'\1"\2"\3:\4', json_text)
                if fixed_text != json_text:
                    fixes_applied.append("Added missing quotes to property names")
                    parsed = json.loads(fixed_text)
                    return {"success": True, "data": parsed, "fixed": True}
            except Exception:
                pass
            
            # Recovery attempt 2: Add missing braces
            try:
                if not json_text.strip().startswith('{') and not json_text.strip().startswith('['):
                    added_open = True
                    json_text = '{' + json_text.strip()
                    fixes_applied.append("Added opening brace")
                else:
                    added_open = False
                    
                if not json_text.strip().endswith('}') and not json_text.strip().endswith(']'):
                    json_text = json_text.strip() + '}'
                    fixes_applied.append("Added closing brace")
                
                parsed = json.loads(json_text)
                return {"success": True, "data": parsed, "fixed": True}
            except Exception:
                # If we added an opening brace but parsing still failed, try with array brackets instead
                if added_open:
                    try:
                        json_text = '[' + json_text[1:].strip()
                        if not json_text.strip().endswith(']'):
                            json_text = json_text.strip()[:-1] + ']'
                        
                        parsed = json.loads(json_text)
                        fixes_applied.append("Converted to JSON array")
                        return {"success": True, "data": parsed, "fixed": True}
                    except Exception:
                        pass
            
            # Recovery attempt 3: Fix single quotes and handle escape characters (common issues)
            try:
                # Step 1: Handle escape sequences and single quotes properly
                import re
                
                # First, we need to properly encode escape characters
                # Replace problematic escape sequences with temporary placeholders
                placeholders = {
                    '\n': '##NEWLINE##',
                    '\r': '##CARRIAGE##',
                    '\t': '##TAB##',
                    '\\': '##BACKSLASH##',
                    '\/': '##FORWARDSLASH##',
                    '\b': '##BACKSPACE##',
                    '\f': '##FORMFEED##'
                }
                
                # Replace escape sequences with placeholders
                temp = json_text
                for escape_seq, placeholder in placeholders.items():
                    temp = temp.replace(escape_seq, placeholder)
                
                # Now safely handle quotes
                # First, temporarily replace already escaped double quotes
                temp = temp.replace('\\"', '##ESCAPED_DQUOTE##')
                temp = temp.replace("\\'", '##ESCAPED_SQUOTE##')
                
                # Replace single quotes with double quotes (that aren't already escaped)
                temp = temp.replace("'", '"')
                
                # Restore escaped quotes with proper JSON escaping
                temp = temp.replace('##ESCAPED_DQUOTE##', '\\"')
                temp = temp.replace('##ESCAPED_SQUOTE##', "\\'")  # This isn't valid JSON but we'll fix it
                
                # Restore placeholders with proper JSON escape sequences
                temp = temp.replace('##NEWLINE##', '\\n')
                temp = temp.replace('##CARRIAGE##', '\\r')
                temp = temp.replace('##TAB##', '\\t')
                temp = temp.replace('##BACKSLASH##', '\\\\')
                temp = temp.replace('##FORWARDSLASH##', '\\/')
                temp = temp.replace('##BACKSPACE##', '\\b')
                temp = temp.replace('##FORMFEED##', '\\f')
                
                # Fix any remaining invalid escapes (convert \' to ' since it's not valid JSON)
                temp = temp.replace("\\'", "'")
                
                if temp != json_text:
                    fixes_applied.append("Fixed quotes and escape sequences")
                    parsed = json.loads(temp)
                    return {"success": True, "data": parsed, "fixed": True}
            except Exception as e:
                logger.error(f"Error fixing quotes and escapes: {e}")
                pass
            
            # Recovery attempt 4: Remove trailing commas (common error)
            try:
                import re
                # Replace trailing commas before closing brackets
                temp = re.sub(r',(\s*[\}\]])', r'\1', json_text)
                
                if temp != json_text:
                    fixes_applied.append("Removed trailing commas")
                    parsed = json.loads(temp)
                    return {"success": True, "data": parsed, "fixed": True}
            except Exception:
                pass
            
            # Recovery attempt 5: Handle unquoted string values
            try:
                import re
                # Find patterns like: "key": value (where value should be quoted)
                # This is complex and might not catch all cases
                temp = re.sub(r':\s*([A-Za-z][A-Za-z0-9_]*)\s*([,\}\]])', r': "\1"\2', json_text)
                
                if temp != json_text:
                    fixes_applied.append("Added quotes to unquoted string values")
                    parsed = json.loads(temp)
                    return {"success": True, "data": parsed, "fixed": True}
            except Exception:
                pass
                
            # All recovery attempts failed
            return {"success": False, "error": str(e)}
    
    # OPERATION: VALIDATE - Just check if it's valid JSON without modifying
    if operation == "validate":
        try:
            json.loads(text)
            result = "✅ The JSON is valid."
        except json.JSONDecodeError as e:
            result = f"❌ The JSON is invalid. Error: {str(e)}"
    
    # OPERATION: PARSE or default - Parse and format the JSON
    elif operation == "parse" or operation is None:
        parse_result = parse_json_with_recovery(text)
        
        if parse_result["success"]:
            parsed_data = parse_result["data"]
            
            # Format the data nicely
            formatted_json = json.dumps(parsed_data, indent=indent)
            
            if "fixed" in parse_result and parse_result["fixed"]:
                result = f"⚠️ Fixed JSON with issues:\n\n```json\n{formatted_json}\n```\n\n"
                if fixes_applied:
                    result += "Fixes applied:\n" + "\n".join([f"- {fix}" for fix in fixes_applied])
            else:
                result = f"✅ Valid JSON parsed successfully:\n\n```json\n{formatted_json}\n```"
        else:
            result = f"❌ Failed to parse JSON: {parse_result['error']}"
    
    # OPERATION: FORMAT - Pretty print the JSON
    elif operation == "format":
        parse_result = parse_json_with_recovery(text)
        
        if parse_result["success"]:
            parsed_data = parse_result["data"]
            
            # Format the data with the specified indentation
            formatted_json = json.dumps(parsed_data, indent=indent, sort_keys=True)
            
            result = f"Formatted JSON:\n\n```json\n{formatted_json}\n```"
            
            if "fixed" in parse_result and parse_result["fixed"]:
                result += "\n\nNote: The original JSON had issues that were fixed before formatting."
                if fixes_applied:
                    result += "\nFixes applied:\n" + "\n".join([f"- {fix}" for fix in fixes_applied])
        else:
            result = f"❌ Failed to format JSON: {parse_result['error']}"
    
    # OPERATION: REPAIR - Attempt to fix malformed JSON
    elif operation == "repair":
        parse_result = parse_json_with_recovery(text)
        
        if parse_result["success"]:
            parsed_data = parse_result["data"]
            
            # Format the data with the specified indentation
            repaired_json = json.dumps(parsed_data, indent=indent)
            
            if "fixed" in parse_result and parse_result["fixed"]:
                result = f"🔧 Successfully repaired the JSON:\n\n```json\n{repaired_json}\n```\n\n"
                if fixes_applied:
                    result += "Repairs performed:\n" + "\n".join([f"- {fix}" for fix in fixes_applied])
            else:
                result = f"ℹ️ The JSON was already valid, but here's the formatted version:\n\n```json\n{repaired_json}\n```"
        else:
            result = f"❌ Failed to repair the JSON: {parse_result['error']}\n\nThe JSON has syntax errors that couldn't be automatically fixed."
    
    # Unknown operation
    else:
        result = f"Error: Unknown operation '{operation}'. Supported operations are: parse, validate, format, repair."
    
    return result

# Mapping from tool names to their implementation functions
TOOL_IMPLEMENTATIONS = {
    "json_tool": json_tool_impl,
    "calculator": calculator_impl,
    "get_weather": weather_impl,
    "search": search_impl,
    "create_plot": create_plot_impl,
    "query_database": query_database_impl,
    # Add alias names that models sometimes use
    "get_current_weather": weather_impl,
    "calculate": calculator_impl,
    "math": calculator_impl,
    "math_tool": calculator_impl,
    "weather": weather_impl,
    "json": json_tool_impl,
    "json_parser": json_tool_impl
}

def tool_playground():
    """Main function for the Tool Playground UI
    
    FIXED VERSION: This version uses buttons for model selection
    and avoids nested expanders.
    """
    st.title("🧰 Tool Playground")
    st.write("Test AI models with function calling and tool integration")
    
    # Initialize session state with defensive programming
    if "tool_chat_history" not in st.session_state:
        st.session_state.tool_chat_history = []
    if "active_tools" not in st.session_state:
        # Set default tools to calculator, json, and weather
        default_tools = ["calculator", "json", "weather"]
        # Filter to only include tools that exist
        st.session_state.active_tools = [tool for tool in default_tools if tool in TOOL_TEMPLATES]
        # Fallback if none of our preferred defaults exist
        if not st.session_state.active_tools and TOOL_TEMPLATES:
            st.session_state.active_tools = [list(TOOL_TEMPLATES.keys())[0]]
    if "custom_tools" not in st.session_state:
        st.session_state.custom_tools = {}
    if "active_mcp_tools" not in st.session_state:
        st.session_state.active_mcp_tools = []
    if "tool_response_hashes" not in st.session_state:
        # Track message hashes to prevent duplicates
        st.session_state.tool_response_hashes = set()
        
    # Helper function to add messages to chat history with duplicate detection
    def add_to_chat_history(message, force=False):
        """Add a message to chat history with duplicate prevention"""
        if not message:
            return False
            
        # For tool responses, calculate a hash to detect duplicates
        is_tool_response = message.get("type") == "tool_response"
        
        if is_tool_response:
            # Create a hash based on content
            import hashlib
            content_hash = hashlib.md5(message.get("content", "").encode()).hexdigest()
            
            # Skip if we've seen this exact content recently
            if content_hash in st.session_state.tool_response_hashes and not force:
                logger.info(f"Skipping duplicate tool response: hash={content_hash[:8]}")
                return False
                
            # Add to our tracking set
            st.session_state.tool_response_hashes.add(content_hash)
            
            # Trim the set if it gets too large (keep most recent 50)
            if len(st.session_state.tool_response_hashes) > 50:
                st.session_state.tool_response_hashes = set(list(st.session_state.tool_response_hashes)[-50:])
        
        # Add the message to the history
        st.session_state.tool_chat_history.append(message)
        return True
        
    # Add to global namespace for use within the function
    st.session_state.add_to_chat_history = add_to_chat_history
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        # Try a direct API call
        try:
            import requests
            api_url = "http://localhost:11434/api/tags"
            response = requests.get(api_url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                if "models" in data and data["models"]:
                    available_models = [model["name"] for model in data["models"] 
                                    if "name" in model and "embed" not in model["name"]]
        except:
            # Fall back to common models
            available_models = ["llama3", "mistral", "mixtral", "phi3", "gemma3", "qwen2"]
    
    # Initialize default model if needed
    if "selected_tool_model" not in st.session_state:
        # Default to a model likely to support tools
        default_models = ["llama3", "mistral", "qwen2", "phi3"]
        for model in default_models:
            if model in available_models:
                st.session_state.selected_tool_model = model
                break
        else:
            # If none of the preferred models are available, select the first one
            st.session_state.selected_tool_model = available_models[0] if available_models else "llama3"
    
    # Filter and sort models to prioritize those with tool capabilities
    tool_capable_models = filter_models_by_capability(available_models, "tools")
    
    # Forced tool-capable model names for better UX
    forced_tool_models = ["llama3", "llama3:7b", "llama3-7b", "llama3.1", "llama3.2", 
                        "mistral", "mistral:7b", "mistral-7b", "mixtral", "phi3",
                        "gemma", "qwen", "qwen2", "openhermes", "yi", "vicuna",
                        "neural-chat", "stablelm", "wizard-math", "mistral-small"]
    
    # Add any model that contains common tool-capable families
    for model in available_models:
        normalized = model.lower()
        if any(family in normalized for family in ["llama", "mistral", "mixtral", "phi", "qwen", "gemma"]):
            if model not in forced_tool_models:
                forced_tool_models.append(model)
    
    # Create prioritized model list
    all_tool_models = []
    
    # First add confirmed tool-capable models
    for model in tool_capable_models:
        if model in available_models and model not in all_tool_models:
            all_tool_models.append(model)
    
    # Then add forced tool models
    for model in forced_tool_models:
        if model in available_models and model not in all_tool_models:
            all_tool_models.append(model)
    
    # Then add remaining models
    for model in available_models:
        if model not in all_tool_models:
            all_tool_models.append(model)
    
    # Main layout: sidebar and content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Tool selection and configuration
        with st.expander("🤖 Model Selection", expanded=True):
            if all_tool_models:
                st.info("Select a model with tool/function calling capabilities")
                
                # Group models by type for better organization
                grouped_models = {}
                for model in all_tool_models:
                    # Determine group based on name patterns
                    if "llama" in model.lower():
                        group = "Llama Models"
                    elif "mistral" in model.lower() or "mixtral" in model.lower():
                        group = "Mistral Models"
                    elif "phi" in model.lower():
                        group = "Phi Models"
                    elif "qwen" in model.lower():
                        group = "Qwen Models"
                    elif "gemma" in model.lower():
                        group = "Gemma Models"
                    else:
                        group = "Other Models"
                    
                    if group not in grouped_models:
                        grouped_models[group] = []
                    grouped_models[group].append(model)
                
                # Create tabs for model families instead of nested expanders
                model_tabs = st.tabs(list(grouped_models.keys()))
                
                # Display models in each tab
                for i, (group, models) in enumerate(grouped_models.items()):
                    with model_tabs[i]:
                        # Create a grid layout for models
                        cols = st.columns(2)  # 2 buttons per row
                        for j, model in enumerate(models):
                            col_idx = j % 2  # alternate between columns
                            
                            # Highlight if model is tool-capable
                            highlight = model in tool_capable_models
                            button_label = f"{model} 🛠️" if highlight else model
                            
                            # Use a unique key for each button
                            with cols[col_idx]:
                                if st.button(button_label, key=f"model_{model}"):
                                    # Set the selected model
                                    st.session_state.selected_tool_model = model
                                    # Log the selection for debugging
                                    logger.info(f"Model selected: {model}")
                                    st.rerun()  # Refresh after changing models
            else:
                st.warning("No models found. Please install some models first.")
            
            # Show current model selection
            if st.session_state.selected_tool_model:
                st.success(f"Current model: **{st.session_state.selected_tool_model}**")
                # Check if model is known to support tool calling
                if is_tools_capable(st.session_state.selected_tool_model):
                    st.info("✅ This model supports tool/function calling")
                else:
                    st.warning("⚠️ This model may not support tool/function calling")
            
            # Model parameters
            temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="tool_temperature"
            )
            
            max_tokens = st.slider(
                "Max Tokens:",
                min_value=1000,
                max_value=32000,
                value=4000,
                step=1000,
                key="tool_max_tokens"
            )
            
            # Recommendation for tool-capable models
            st.info("For best results with tools, try using models like llama3, mistral, qwen2, or phi3.")
        
        # Tool management section continues unchanged
        with st.expander("🧰 Tool Management", expanded=True):
            # Available predefined tools - now in a tab layout for cleaner UI
            tool_tabs = st.tabs(["Predefined Tools", "Custom Tool", "MCP Tools"])
            
            with tool_tabs[0]:
                # Multiselect for selecting predefined tools
                # Ensure default values are in the options list
                available_tool_options = list(TOOL_TEMPLATES.keys())
                valid_defaults = [tool for tool in st.session_state.active_tools if tool in available_tool_options]
                
                selected_predefined_tools = st.multiselect(
                    "Select tools to use:",
                    available_tool_options,
                    default=valid_defaults,
                    key="tool_selector"
                )
            
            with tool_tabs[1]:
                # Input for custom tool schema
                custom_tool_schema = st.text_area(
                    "Define a custom tool (JSON schema):",
                    value="""{"type": "function", "function": {"name": "example_tool", "description": "An example tool", "parameters": {"type": "object", "properties": {"input": {"type": "string", "description": "An input parameter"}}, "required": ["input"]}}}""",
                    height=150,
                    key="custom_tool_schema"
                )
            
                # Add button for the custom tool
                if st.button("Add Custom Tool"):
                    try:
                        # Parse the JSON schema
                        custom_tool = json.loads(custom_tool_schema)
                        
                        # Validate the schema
                        if not isinstance(custom_tool, dict) or "type" not in custom_tool or "function" not in custom_tool:
                            st.error("Invalid tool schema. Must include 'type' and 'function' fields.")
                        elif "name" not in custom_tool["function"] or "parameters" not in custom_tool["function"]:
                            st.error("Invalid function schema. Must include 'name' and 'parameters' fields.")
                        else:
                            # Add the custom tool to the session state
                            tool_name = custom_tool["function"]["name"]
                            st.session_state.custom_tools[tool_name] = custom_tool
                            st.success(f"Custom tool '{tool_name}' added successfully!")
                            
                            # Add the tool to the active tools list if not already present
                            if tool_name not in st.session_state.active_tools:
                                st.session_state.active_tools.append(tool_name)
                                st.rerun()
                    except json.JSONDecodeError:
                        st.error("Invalid JSON. Please check your schema.")
                        
                # Display and manage custom tools
                if st.session_state.custom_tools:
                    st.markdown("### Manage Custom Tools")
                    st.markdown("---")
                    
                    # Create columns for better organization
                    for tool_name, tool_schema in st.session_state.custom_tools.items():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Tool: {tool_name}**")
                            st.json(tool_schema)
                        
                        with col2:
                            st.markdown("&nbsp;")  # Add space for alignment
                            if st.button(f"Remove {tool_name}", key=f"remove_{tool_name}"):
                                # Remove the tool from the session state
                                del st.session_state.custom_tools[tool_name]
                                
                                # Remove the tool from the active tools list if present
                                if tool_name in st.session_state.active_tools:
                                    st.session_state.active_tools.remove(tool_name)
                                
                                st.success(f"Custom tool '{tool_name}' removed successfully!")
                                st.rerun()
                        
                        st.markdown("---")  # Separator between tools
            
            # MCP Tools Integration
            with tool_tabs[2]:
                if st.button("Refresh MCP Tools"):
                    # Rediscover MCP tools
                    try:
                        mcp_tools, mcp_schemas = get_available_mcp_tools()
                        st.session_state.mcp_tools = mcp_tools
                        st.session_state.mcp_schemas = mcp_schemas
                        st.success("MCP tools refreshed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error refreshing MCP tools: {str(e)}")
                        st.session_state.mcp_tools = {}
                        st.session_state.mcp_schemas = []
                
                if "mcp_tools" in st.session_state and st.session_state.mcp_tools:
                    st.write("These are MCP tools found on your local system")
                    
                    # Display available MCP tools with checkboxes
                    mcp_tool_options = list(st.session_state.mcp_tools.keys())
                    # Ensure default values exist in options
                    valid_mcp_defaults = [tool for tool in st.session_state.active_mcp_tools if tool in mcp_tool_options]
                    
                    selected_mcp_tools = st.multiselect(
                        "Select MCP tools to use:",
                        mcp_tool_options,
                        default=valid_mcp_defaults,
                        key="mcp_tool_selector"
                    )
                
                    # Show details about selected MCP tools
                    if selected_mcp_tools:
                        st.markdown("### Selected MCP Tools")
                        
                        # Display each selected tool in a cleaner format without nested expanders
                        for tool_name in selected_mcp_tools:
                            st.markdown(f"#### Tool: {tool_name}")
                            tool_info = st.session_state.mcp_tools[tool_name]
                            
                            # Create columns for better organization
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**Type:** {tool_info['type']}")
                            
                            with col2:
                                st.markdown(f"**Path:** {tool_info['path']}")
                            
                            # Display additional information based on tool type
                            if tool_info["type"] == "json" and "manifest" in tool_info:
                                if "description" in tool_info["manifest"]:
                                    st.markdown(f"**Description:** {tool_info['manifest']['description']}")
                                if "schema" in tool_info["manifest"]:
                                    with st.container():
                                        st.markdown("**Schema:**")
                                        st.json(tool_info["manifest"]["schema"])
                            
                            elif tool_info["type"] in ["python", "executable"] and "info" in tool_info:
                                if "description" in tool_info["info"]:
                                    st.markdown(f"**Description:** {tool_info['info']['description']}")
                                if "schema" in tool_info["info"]:
                                    with st.container():
                                        st.markdown("**Schema:**")
                                        st.json(tool_info["info"]["schema"])
                            
                            st.markdown("---")
                    
                    # Update active MCP tools list
                    st.session_state.active_mcp_tools = selected_mcp_tools
                else:
                    st.info("No MCP tools found on this system. Add tools to an MCP folder in your Documents directory.")
                    
                    # Create example MCP tool (this part remains unchanged)
                    if st.button("Create Example MCP Tool"):
                        import os
                        
                        # Create directories if they don't exist
                        mcp_dir = os.path.expanduser("~/Documents/Cline/MCP")
                        os.makedirs(mcp_dir, exist_ok=True)
                        
                        example_tool_dir = os.path.join(mcp_dir, "example_tool")
                        os.makedirs(example_tool_dir, exist_ok=True)
                        
                        # Create manifest.json
                        manifest_content = {
                            "name": "example_tool",
                            "description": "An example MCP tool for demonstration",
                            "version": "1.0",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "The text to process"
                                    },
                                    "operation": {
                                        "type": "string",
                                        "enum": ["uppercase", "lowercase", "reverse"],
                                        "description": "The operation to perform on the text"
                                    }
                                },
                                "required": ["text", "operation"]
                            },
                            "executable": "example_tool.py"
                        }
                        
                        with open(os.path.join(example_tool_dir, "manifest.json"), "w") as f:
                            json.dump(manifest_content, f, indent=2)
                        
                        # Create example_tool.py
                        tool_script = '''#!/usr/bin/env python3
import json
import sys

def process_text(text, operation):
    """Process text based on the specified operation."""
    if operation == "uppercase":
        return text.upper()
    elif operation == "lowercase":
        return text.lower()
    elif operation == "reverse":
        return text[::-1]
    else:
        return f"Unknown operation: {operation}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check if this is an info request
        if sys.argv[1] == "--info":
            # Return tool info as JSON
            info = {
                "name": "example_tool",
                "description": "An example MCP tool that processes text",
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to process"
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["uppercase", "lowercase", "reverse"],
                            "description": "The operation to perform on the text"
                        }
                    },
                    "required": ["text", "operation"]
                }
            }
            print(json.dumps(info))
            sys.exit(0)
        
        # Check if this is an execution request
        if sys.argv[1] == "--execute" and len(sys.argv) > 2:
            try:
                # Parse the arguments
                args = json.loads(sys.argv[2])
                
                # Extract parameters
                text = args.get("text", "")
                operation = args.get("operation", "")
                
                # Process the text
                result = process_text(text, operation)
                
                # Return the result as JSON
                print(json.dumps({
                    "result": result,
                    "original_text": text,
                    "operation": operation
                }))
                sys.exit(0)
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON arguments"}))
                sys.exit(1)
            except Exception as e:
                print(json.dumps({"error": str(e)}))
                sys.exit(1)
    
    # Return error for invalid usage
    print(json.dumps({"error": "Invalid arguments. Use --info or --execute"}))
    sys.exit(1)
'''
                        
                        with open(os.path.join(example_tool_dir, "example_tool.py"), "w") as f:
                            f.write(tool_script)
                        
                        # Make the script executable
                        os.chmod(os.path.join(example_tool_dir, "example_tool.py"), 0o755)
                        
                        st.success("Example MCP tool created! Click 'Refresh MCP Tools' to see it.")
                        
                        # Try to rediscover MCP tools
                        try:
                            mcp_tools, mcp_schemas = get_available_mcp_tools()
                            st.session_state.mcp_tools = mcp_tools
                            st.session_state.mcp_schemas = mcp_schemas
                        except:
                            pass
            
            # Update active tools list based on selection
            st.session_state.active_tools = selected_predefined_tools + list(st.session_state.custom_tools.keys())
        
        # Test prompt templates
        with st.expander("📝 Test Prompts", expanded=False):
            st.subheader("Sample Prompts")
            
            prompts = {
                "calculator": "What is 342 multiplied by 15?",
                "weather": "What's the weather like in Tokyo?",
                "search": "Search for information about artificial intelligence.",
                "plot": "Create a bar chart showing sales data for the last 5 months.",
                "database": "Query the users table and show me the first 3 records.",
                "json": "Can you parse and repair this JSON for me? {id: 1, name: 'John Doe', active: true, skills: ['javascript', 'python']}",
                "mcp_example": "Use the example tool to convert 'Hello World' to uppercase."
            }
            
            for tool, prompt in prompts.items():
                # Special case for MCP tools
                if tool == "mcp_example" and "example_tool" in st.session_state.active_mcp_tools:
                    if st.button(f"Use: {prompt}", key=f"prompt_{tool}"):
                        st.session_state.selected_tool_prompt = prompt
                        st.rerun()
                # Standard tools
                elif tool in st.session_state.active_tools or tool in st.session_state.custom_tools:
                    if st.button(f"Use: {prompt}", key=f"prompt_{tool}"):
                        st.session_state.selected_tool_prompt = prompt
                        st.rerun()
    
    with col2:
        # Chat interface
        st.subheader("Chat with Tools")
        
        # Display currently enabled tools
        if st.session_state.active_tools:
            st.info(f"Enabled tools: {', '.join(st.session_state.active_tools)}")
        else:
            st.warning("No tools are currently enabled. Please select at least one tool.")
        
        # Display current selected model
        st.info(f"Using model: {st.session_state.selected_tool_model}")
        
        # Chat history display
        for message in st.session_state.tool_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If this is a tool call, show the tool call details
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        with st.expander(f"Tool Call: {tool_call['function']['name']}"):
                            st.json(tool_call["function"]["arguments"])
                
                # If this is a tool response, show it in a different style
                if message.get("type") == "tool_response":
                    st.info(message["content"])
        
        # Input for new message
        user_input = st.chat_input("Ask a question using tools...", key="tool_prompt")
        
        # Use the test prompt if one was selected
        if hasattr(st.session_state, "selected_tool_prompt") and st.session_state.selected_tool_prompt:
            user_input = st.session_state.selected_tool_prompt
            st.session_state.selected_tool_prompt = None
        
        if user_input:
            # Check if this is a simple calculator operation before attempting complex tool calling
            # Comprehensive patterns to detect calculation queries
            simple_calculator_patterns = [
                # Multiplication
                r"what is (\d+) (?:multiplied by|times|x) (\d+)",
                r"(\d+) (?:multiplied by|times|x) (\d+)",
                r"calculate (\d+) [\*x] (\d+)",
                r"(\d+) [\*x] (\d+)",
                
                # Addition
                r"what is (\d+) (?:plus|\+|added to) (\d+)",
                r"(\d+) (?:plus|\+|added to) (\d+)",
                
                # Subtraction
                r"what is (\d+) (?:minus|\-|subtracted by) (\d+)",
                r"(\d+) (?:minus|\-|subtracted by) (\d+)",
                
                # Division
                r"what is (\d+) (?:divided by|÷|/) (\d+)",
                r"(\d+) (?:divided by|÷|/) (\d+)",
                
                # General arithmetic
                r"what is (\d+) [+\-*/] (\d+)",
                r"(\d+) [+\-*/] (\d+)",
                
                # General calculation question
                r"calculate (\d+\s*[\+\-\*\/]\s*\d+)",
                r"what is (\d+\s*[\+\-\*\/]\s*\d+)",
                
                # Numbers in context without explicit operator
                r"what is the (?:product|result|answer|total|sum|quotient) of (\d+) and (\d+)"
            ]
            
            is_simple_calculation = False
            for pattern in simple_calculator_patterns:
                import re
                match = re.search(pattern, user_input.lower())
                if match:
                    # This looks like a simple calculation, handle it directly
                    logger.info(f"Detected simple calculation: {user_input}")
                    is_simple_calculation = True
                    
                    # Add user message to chat history
                    st.session_state.add_to_chat_history({"role": "user", "content": user_input})
                    
                    # Process directly with calculator
                    try:
                        result = TOOL_IMPLEMENTATIONS["calculator"](operation=user_input)
                        st.session_state.add_to_chat_history({
                            "role": "assistant",
                            "content": result
                        })
                        st.rerun()
                        break
                    except Exception as e:
                        logger.error(f"Error in direct calculator handling: {e}")
                        # Fall back to standard processing
                        is_simple_calculation = False
            
            if not is_simple_calculation:
                # Add user message to chat history
                st.session_state.add_to_chat_history({"role": "user", "content": user_input})
            
            # Prepare tools for the model
            tools = []
            for tool_name in st.session_state.active_tools:
                if tool_name in TOOL_TEMPLATES:
                    tools.append(TOOL_TEMPLATES[tool_name])
                elif tool_name in st.session_state.custom_tools:
                    tools.append(st.session_state.custom_tools[tool_name])
            
            # Add MCP tools
            if "mcp_schemas" in st.session_state:
                for tool_name in st.session_state.active_mcp_tools:
                    # Find matching schema
                    for schema in st.session_state.mcp_schemas:
                        if schema["function"]["name"] == f"mcp__{tool_name}":
                            tools.append(schema)
                            break
            
            # Display typing indicator
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                try:
                    # Prepare messages for the model
                    messages = []
                    for msg in st.session_state.tool_chat_history:
                        if msg["role"] == "user":
                            messages.append({"role": "user", "content": msg["content"]})
                        elif msg["role"] == "assistant" and "tool_calls" not in msg:
                            messages.append({"role": "assistant", "content": msg["content"]})
                        elif msg["role"] == "assistant" and "tool_calls" in msg:
                            tool_message = {"role": "assistant", "content": msg["content"], "tool_calls": msg["tool_calls"]}
                            messages.append(tool_message)
                        elif msg.get("type") == "tool_response":
                            messages.append({"role": "tool", "content": msg["content"], "tool_call_id": msg["tool_call_id"]})
                    
                    # Check if model is known to support tools before sending request
                    try:
                        if not is_tools_capable(st.session_state.selected_tool_model):
                            # Model is not in our tools-capable list, show a warning with more information
                            logger.warning(f"Model '{st.session_state.selected_tool_model}' is not known to support tools. Attempting anyway but may fail.")
                            
                            warning_msg = f"""
                            ⚠️ **Important:** Model '{st.session_state.selected_tool_model}' is not officially known to support tools.
                            
                            This may result in an error. For best results, use a model designed for function calling:
                            - llama3 (best support)
                            - mistral
                            - qwen
                            - phi3
                            
                            Documentation: [Ollama Function Calling Models](https://ollama.com/search?c=tools)
                            """
                            st.warning(warning_msg)
                    except Exception as check_error:
                        # If the capability check fails, log it but continue
                        logger.error(f"Error checking tool capability: {check_error}")
                        # Don't show a warning to avoid confusing the user
                    
                    # Use the model name directly from session state
                    model_to_use = st.session_state.selected_tool_model
                    
                    # Save the model specifically for this request to ensure consistency
                    # This prevents other UI components from changing the model mid-request
                    st.session_state.last_tool_call_model = model_to_use
                    
                    # DEBUG: Log which model we're actually using for better diagnostics
                    logger.info(f"Using model for tool call: {model_to_use}")
                    
                    # Double-check that it's actually installed
                    try:
                        available = get_available_models()
                        if model_to_use not in available:
                            logger.warning(f"Model {model_to_use} not found in available models: {available}")
                            # Fall back to a model that exists
                            for fallback in ['llama3', 'mistral', 'mixtral']:
                                if fallback in available:
                                    logger.info(f"Falling back to model: {fallback}")
                                    model_to_use = fallback
                                    break
                    except Exception as model_check_error:
                        logger.error(f"Error checking available models: {model_check_error}")
                    
                    # Send request to model
                    client = ollama.Client()
                    
                    # Add debug logging to show what we're sending
                    logger.info(f"Sending request to model: {model_to_use}")
                    logger.info(f"Tools: {json.dumps(tools[:2], indent=2)}..." if tools else "No tools")
                    
                    # Make the API call
                    # Sanitize objects to ensure JSON serialization works
                    safe_messages = sanitize_for_json(messages)
                    safe_tools = sanitize_for_json(tools)
                    
                    response = client.chat(
                        model=model_to_use,
                        messages=safe_messages,
                        tools=safe_tools,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    )
                    
                    # Log the response for debugging (sanitized)
                    try:
                        logger.info(f"Received response from model: {json.dumps(sanitize_for_json(response.get('message', {})), indent=2)}")
                    except Exception as log_err:
                        logger.error(f"Error logging response: {log_err}")
                        logger.info(f"Response received but could not be serialized")
                    
                    # Handle the response
                    if "message" in response:
                        assistant_message = response["message"]
                        
                        # Look for potential JSON content in the raw response that should have been a tool call
                        content = assistant_message.get("content", "")
                        should_use_json_tool = False
                        json_text = None
                        
                        # Check if this response contains JSON that should be processed with the JSON tool
                        if "json_tool" in [tool.get("function", {}).get("name", "") for tool in tools]:
                            import re
                            # Look for JSON patterns in the response (either object or array)
                            json_patterns = [
                                # Code block patterns
                                r'```(?:json)?\s*(\{.*?\})\s*```',     # JSON objects in code blocks
                                r'```(?:json)?\s*(\[.*?\])\s*```',     # JSON arrays in code blocks
                                
                                # Plain JSON patterns with double quotes
                                r'(\{\s*"[^"]+"\s*:.*?\})',            # Plain JSON objects with double quotes
                                r'(\[\s*\{.*?\}\s*\])',                # Plain JSON arrays
                                
                                # Common invalid JSON patterns (single quotes or missing quotes)
                                r'(\{\s*id\s*:.*?skills\s*:.*?\})',    # The specific pattern from the test case
                                r'(\{\s*\'[^\']+\'\s*:.*?\})',         # JSON objects with single quotes
                                r'(\{\s*\w+\s*:.*?\})',                # JSON objects with unquoted keys
                            ]
                            
                            for pattern in json_patterns:
                                match = re.search(pattern, content, re.DOTALL)
                                if match:
                                    logger.info("Detected JSON in model response that should use the JSON tool")
                                    json_text = match.group(1)
                                    should_use_json_tool = True
                                    
                                    # Check if the query was explicitly about JSON repair/parsing
                                    last_user_message = ""
                                    for msg in st.session_state.tool_chat_history:
                                        if msg["role"] == "user":
                                            last_user_message = msg["content"]
                                    
                                    json_related_terms = ["json", "parse", "repair", "validate", "format"]
                                    if any(term in last_user_message.lower() for term in json_related_terms):
                                        logger.info("User query was explicitly about JSON processing")
                                        break
                        
                        # Update the message placeholder with the response content
                        message_placeholder.markdown(content)
                        
                        # Add the assistant message to the chat history
                        assistant_history_msg = {
                            "role": "assistant",
                            "content": content
                        }
                        
                        # Handle tool calls if present
                        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                            assistant_history_msg["tool_calls"] = assistant_message["tool_calls"]
                            
                            # Process each tool call
                            for tool_call in assistant_message["tool_calls"]:
                                function_name = tool_call["function"]["name"]
                                
                                # Parse the arguments
                                try:
                                    # Check if arguments is already a dict or needs to be parsed from JSON string
                                    arguments = tool_call["function"]["arguments"]
                                    logger.info(f"Raw arguments for {function_name}: {arguments} (type: {type(arguments)})")
                                    
                                    # Better argument parsing with more extensive error reporting
                                    if isinstance(arguments, str):
                                        try:
                                            # Try to parse as JSON
                                            arguments = json.loads(arguments)
                                            logger.info(f"Parsed arguments from JSON string: {arguments}")
                                        except json.JSONDecodeError as e:
                                            logger.error(f"JSON parsing error: {e}")
                                            # Try to extract key-value pairs from non-JSON string
                                            try:
                                                # Check for simple key-value format like "a: 342, b: 15"
                                                import re
                                                parsed_args = {}
                                                
                                                # Patterns to match different formats
                                                patterns = [
                                                    r'(\w+)\s*[:=]\s*([^,}]+)',  # key: value or key = value
                                                    r'"(\w+)"\s*[:=]\s*([^,}]+)',  # "key": value
                                                    r"'(\w+)'\s*[:=]\s*([^,}]+)"   # 'key': value
                                                ]
                                                
                                                for pattern in patterns:
                                                    matches = re.findall(pattern, arguments)
                                                    for key, value in matches:
                                                        # Convert to appropriate types
                                                        try:
                                                            # Try to convert to number if possible
                                                            if value.strip().isdigit():
                                                                parsed_args[key.strip()] = int(value.strip())
                                                            elif re.match(r'^-?\d+(\.\d+)?$', value.strip()):
                                                                parsed_args[key.strip()] = float(value.strip())
                                                            else:
                                                                # Remove quotes if present
                                                                val = value.strip()
                                                                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                                                                    val = val[1:-1]
                                                                parsed_args[key.strip()] = val
                                                        except ValueError:
                                                            parsed_args[key.strip()] = value.strip()
                                                
                                                # If we found any arguments, use them
                                                if parsed_args:
                                                    logger.info(f"Extracted arguments from string: {parsed_args}")
                                                    arguments = parsed_args
                                                else:
                                                    # If all else fails, just use the string as the operation
                                                    logger.info(f"Using string as operation: {arguments}")
                                                    arguments = {"operation": arguments}
                                            except Exception as parse_error:
                                                logger.error(f"Failed to parse arguments from string: {parse_error}")
                                                arguments = {"operation": arguments}
                                    
                                    # Extract arguments from direct text for calculator
                                    if function_name == "calculator" and (not arguments or len(arguments) == 0 or all(v is None for v in arguments.values())):
                                        # Check tool_call for any usable content
                                        logger.info(f"No valid arguments found, checking for direct content in tool_call: {tool_call}")
                                        
                                        # Try to use any available text
                                        if "content" in tool_call and tool_call["content"]:
                                            arguments = {"operation": tool_call["content"]}
                                        elif "description" in tool_call and tool_call["description"]:
                                            arguments = {"operation": tool_call["description"]}
                                            
                                    # Log final arguments
                                    logger.info(f"Final arguments for {function_name}: {arguments}")
                                except Exception as arg_error:
                                    logger.error(f"Failed to parse arguments: {arg_error}")
                                    arguments = {"error": "Failed to parse arguments"}
                                
                                # Execute the tool call
                                if function_name in TOOL_IMPLEMENTATIONS:
                                    # If this is a calculator with missing arguments, try to use the original user query
                                    if function_name == "calculator" and (not arguments or all(arg is None for arg in arguments.values())):
                                        logger.warning(f"Calculator called with empty arguments, attempting to use direct input")
                                        
                                        # Find the user's original query from chat history
                                        user_query = None
                                        for msg in st.session_state.tool_chat_history:
                                            if msg["role"] == "user":
                                                user_query = msg["content"]
                                        
                                        if user_query:
                                            logger.info(f"Using direct user query for calculator: {user_query}")
                                            # Call calculator with the raw user query as the operation
                                            result = TOOL_IMPLEMENTATIONS[function_name](operation=user_query)
                                            
                                            # Check if we've already shown this exact result to avoid duplication
                                            result_shown = False
                                            for msg in st.session_state.tool_chat_history[-3:]:  # Check last few messages
                                                if msg.get("content") == result:
                                                    result_shown = True
                                                    logger.info(f"Avoiding duplicate result display: {result}")
                                                    break
                                            
                                            if not result_shown:
                                                # Add the tool response to the chat history with duplicate prevention
                                                tool_response = {
                                                    "role": "assistant",
                                                    "type": "tool_response",
                                                    "content": result,
                                                    "tool_call_id": tool_call.get("id", str(uuid.uuid4()))
                                                }
                                                
                                                if st.session_state.add_to_chat_history(tool_response):
                                                    # Only rerun if we actually added a new response (not a duplicate)
                                                    # Force an immediate UI update to show the tool response
                                                    st.rerun()
                                            continue  # Skip to the next tool call
                                    try:
                                        # Call the implementation function with the parsed arguments
                                        result = TOOL_IMPLEMENTATIONS[function_name](**arguments)
                                        
                                        # Check if the tool call has an id field and handle the case if it doesn't
                                        if "id" not in tool_call:
                                            logger.error(f"Tool call missing 'id' field: {tool_call}")
                                            # Generate a random id as fallback
                                            import uuid
                                            tool_call_id = str(uuid.uuid4())
                                            logger.info(f"Generated fallback ID: {tool_call_id}")
                                        else:
                                            tool_call_id = tool_call["id"]
                                        
                                        # Check if we've already shown this exact result to avoid duplication
                                        result_shown = False
                                        for msg in st.session_state.tool_chat_history[-3:]:  # Check last few messages
                                            if msg.get("content") == result:
                                                result_shown = True
                                                logger.info(f"Avoiding duplicate result display: {result}")
                                                break
                                        
                                        if not result_shown:
                                            # Add the tool response to the chat history with duplicate prevention
                                            tool_response = {
                                                "role": "assistant",
                                                "type": "tool_response", 
                                                "content": result,
                                                "tool_call_id": tool_call_id
                                            }
                                            
                                            if st.session_state.add_to_chat_history(tool_response):
                                                # Only rerun if we actually added a new message (not a duplicate)
                                                # Force an immediate UI update to show the tool response
                                                st.rerun()
                                    except Exception as e:
                                        # Handle errors in tool execution
                                        error_message = f"Error executing tool {function_name}: {str(e)}"
                                        # Ensure tool_call_id exists
                                        if "id" not in tool_call:
                                            import uuid
                                            tool_call_id = str(uuid.uuid4())
                                            logger.info(f"Generated fallback ID for error response: {tool_call_id}")
                                        else:
                                            tool_call_id = tool_call["id"]
                                            
                                        # Check for duplicate error messages
                                        error_shown = False
                                        for msg in st.session_state.tool_chat_history[-3:]:
                                            if msg.get("content") == error_message:
                                                error_shown = True
                                                logger.info(f"Avoiding duplicate error message: {error_message}")
                                                break
                                                
                                        if not error_shown:
                                            st.session_state.tool_chat_history.append({
                                                "role": "assistant",
                                                "type": "tool_response",
                                                "content": error_message,
                                                "tool_call_id": tool_call_id
                                            })
                                # Check if this is an MCP tool call
                                elif function_name.startswith("mcp__"):
                                    try:
                                        # Extract the real tool name (without the mcp__ prefix)
                                        mcp_tool_name = function_name[5:]
                                        logger.info(f"Executing MCP tool: {mcp_tool_name}")
                                        
                                        # Execute the MCP tool
                                        result = execute_mcp_tool(mcp_tool_name, st.session_state.mcp_tools, arguments)
                                        
                                        # Check if the tool call has an id field and handle the case if it doesn't
                                        if "id" not in tool_call:
                                            logger.error(f"Tool call missing 'id' field: {tool_call}")
                                            # Generate a random id as fallback
                                            import uuid
                                            tool_call_id = str(uuid.uuid4())
                                            logger.info(f"Generated fallback ID: {tool_call_id}")
                                        else:
                                            tool_call_id = tool_call["id"]
                                        
                                        # Check if we've already shown this exact result to avoid duplication
                                        result_shown = False
                                        for msg in st.session_state.tool_chat_history[-3:]:  # Check last few messages
                                            if msg.get("content") == result:
                                                result_shown = True
                                                logger.info(f"Avoiding duplicate result display: {result}")
                                                break
                                        
                                        if not result_shown:
                                            # Add the tool response to the chat history with duplicate prevention
                                            tool_response = {
                                                "role": "assistant",
                                                "type": "tool_response", 
                                                "content": result,
                                                "tool_call_id": tool_call_id
                                            }
                                            
                                            if st.session_state.add_to_chat_history(tool_response):
                                                # Only rerun if we actually added a new message (not a duplicate)
                                                # Force an immediate UI update to show the MCP tool response
                                                st.rerun()
                                    except Exception as e:
                                        # Handle errors in MCP tool execution
                                        error_message = f"Error executing MCP tool {function_name}: {str(e)}"
                                        # Ensure tool_call_id exists
                                        if "id" not in tool_call:
                                            import uuid
                                            tool_call_id = str(uuid.uuid4())
                                            logger.info(f"Generated fallback ID for error response: {tool_call_id}")
                                        else:
                                            tool_call_id = tool_call["id"]
                                            
                                        # Check for duplicate error messages
                                        error_shown = False
                                        for msg in st.session_state.tool_chat_history[-3:]:
                                            if msg.get("content") == error_message:
                                                error_shown = True
                                                logger.info(f"Avoiding duplicate error message: {error_message}")
                                                break
                                                
                                        if not error_shown:
                                            st.session_state.tool_chat_history.append({
                                                "role": "assistant",
                                                "type": "tool_response",
                                                "content": error_message,
                                                "tool_call_id": tool_call_id
                                            })
                                else:
                                    # Unknown tool
                                    error_message = f"Tool {function_name} is not implemented"
                                    # Ensure tool_call_id exists
                                    if "id" not in tool_call:
                                        import uuid
                                        tool_call_id = str(uuid.uuid4())
                                        logger.info(f"Generated fallback ID for unknown tool: {tool_call_id}")
                                    else:
                                        tool_call_id = tool_call["id"]
                                        
                                    st.session_state.tool_chat_history.append({
                                        "role": "assistant",
                                        "type": "tool_response",
                                        "content": error_message,
                                        "tool_call_id": tool_call_id
                                    })
                            
                            # Send a follow-up request with the tool responses
                            follow_up_messages = messages.copy()
                            
                            # Add the assistant message with tool calls
                            follow_up_messages.append({
                                "role": "assistant",
                                "content": assistant_message.get("content", ""),
                                "tool_calls": assistant_message["tool_calls"]
                            })
                            
                            # Add the tool responses
                            sent_tools = False
                            for msg in st.session_state.tool_chat_history:
                                if msg.get("type") == "tool_response":
                                    try:
                                        # Try to match by tool_call_id (standard way)
                                        if any(msg["tool_call_id"] == (tc.get("id", "") or "") for tc in assistant_message["tool_calls"]):
                                            follow_up_messages.append({
                                                "role": "tool",
                                                "content": msg["content"],
                                                "tool_call_id": msg["tool_call_id"]
                                            })
                                            sent_tools = True
                                    except Exception as e:
                                        logger.error(f"Error matching tool responses: {e}")
                            
                            # If we didn't add any tool responses, try a different approach for models that don't follow the standard
                            if not sent_tools and st.session_state.tool_chat_history:
                                # Find the most recent tool response
                                recent_responses = [msg for msg in st.session_state.tool_chat_history if msg.get("type") == "tool_response"]
                                if recent_responses:
                                    most_recent = recent_responses[-1]
                                    # Add it as a generic tool response
                                    logger.info(f"Using fallback approach for tool response: {most_recent['content']}")
                                    
                                    try:
                                        # Try to get any tool_call_id from the message
                                        tool_call_id = most_recent.get("tool_call_id", "")
                                        if not tool_call_id and assistant_message.get("tool_calls") and len(assistant_message["tool_calls"]) > 0:
                                            tool_call_id = assistant_message["tool_calls"][0].get("id", "generic_id")
                                        
                                        follow_up_messages.append({
                                            "role": "tool",
                                            "content": most_recent["content"],
                                            "tool_call_id": tool_call_id
                                        })
                                    except Exception as e:
                                        logger.error(f"Error adding fallback tool response: {e}")
                                        # Last resort - add a simple message
                                        follow_up_messages.append({
                                            "role": "user",
                                            "content": f"The result was: {most_recent['content']}"
                                        })
                            
                            # Send the follow-up request
                            # Log what we're sending for the follow-up request
                            logger.info(f"Sending follow-up request with tool results to model: {model_to_use}")
                            logger.info(f"Follow-up messages: {json.dumps(follow_up_messages[-2:], indent=2)}...")
                            
                            # Sanitize follow-up messages and tools
                            safe_followup_messages = sanitize_for_json(follow_up_messages)
                            
                            # Before trying a follow-up request, check if the result is already satisfactory
                            # Look for any successful tool responses already shown to the user
                            result_already_shown = False
                            for msg in st.session_state.tool_chat_history[-3:]:
                                # Check for calculator results
                                if msg.get("type") == "tool_response" and "result" in msg.get("content", "").lower():
                                    logger.info(f"Found calculator result in tool response: {msg.get('content')}")
                                    result_already_shown = True
                                    break
                                # Check for weather results - handle capitalized "Weather" too
                                elif msg.get("type") == "tool_response" and ("weather for" in msg.get("content", "").lower() or "Weather for" in msg.get("content", "")):
                                    logger.info(f"Found weather result in tool response: {msg.get('content')}")
                                    result_already_shown = True
                                    break
                                # Check for search results
                                elif msg.get("type") == "tool_response" and "search results for:" in msg.get("content", "").lower():
                                    logger.info(f"Found search result in tool response")
                                    result_already_shown = True
                                    break
                                # Check for database query results
                                elif msg.get("type") == "tool_response" and "query results from table" in msg.get("content", "").lower():
                                    logger.info(f"Found database query result in tool response")
                                    result_already_shown = True
                                    break
                                # Check for JSON tool results 
                                elif msg.get("type") == "tool_response" and any(x in msg.get("content", "").lower() for x in ["json is valid", "parsed json", "json parsed", "repaired json", "formatted json"]):
                                    logger.info(f"Found JSON result in tool response")
                                    result_already_shown = True
                                    break
                                    
                            if result_already_shown:
                                logger.info("Skipping follow-up request as result is already shown")
                                # Create a simple acknowledgment response
                                follow_up_response = {
                                    "message": {
                                        "content": "",  # Empty content to avoid duplication
                                        "role": "assistant"
                                    }
                                }
                            else:
                                # Only make the follow-up request if needed
                                try:
                                    follow_up_response = client.chat(
                                        model=st.session_state.last_tool_call_model or model_to_use,  # Ensure consistent model between requests
                                        messages=safe_followup_messages,
                                        tools=safe_tools,  # Use the already sanitized tools
                                        options={
                                            "temperature": temperature,
                                            "num_predict": max_tokens
                                        }
                                    )
                                except Exception as follow_up_error:
                                    logger.error(f"Error during follow-up request: {follow_up_error}")
                                    # Create a synthetic response based on the tool results
                                
                                    # Find tool response content
                                    tool_results = []
                                    for msg in follow_up_messages:
                                        if msg.get("role") == "tool":
                                            tool_results.append(msg.get("content", ""))
                                    
                                    # Generate a simple summary message
                                    if tool_results:
                                        summary = tool_results[0] if len(tool_results) == 1 else "Multiple results: " + "; ".join(tool_results)
                                        follow_up_response = {
                                            "message": {
                                                "content": f"The calculation result is: {summary}",
                                                "role": "assistant"
                                            }
                                        }
                                    else:
                                        follow_up_response = {
                                            "message": {
                                                "content": "I couldn't perform the calculation at this time.",
                                                "role": "assistant"
                                            }
                                        }
                            
                            # Log the follow-up response (sanitized)
                            try:
                                logger.info(f"Received follow-up response: {json.dumps(sanitize_for_json(follow_up_response.get('message', {})), indent=2)}")
                            except Exception as log_err:
                                logger.error(f"Error logging follow-up response: {log_err}")
                                logger.info(f"Follow-up response received but could not be serialized")
                            
                            # Add the follow-up response to the chat history
                            try:
                                if "message" in follow_up_response and "content" in follow_up_response["message"]:
                                    follow_up_content = follow_up_response["message"]["content"]
                                    st.session_state.add_to_chat_history({
                                        "role": "assistant",
                                        "content": follow_up_content
                                    })
                                elif isinstance(follow_up_response, dict):
                                    # Try to extract content from any recognizable format
                                    content = None
                                    if "response" in follow_up_response:
                                        content = follow_up_response["response"]
                                    elif "result" in follow_up_response:
                                        content = follow_up_response["result"]
                                    elif "content" in follow_up_response:
                                        content = follow_up_response["content"]
                                    
                                    if content:
                                        st.session_state.add_to_chat_history({
                                            "role": "assistant",
                                            "content": content
                                        })
                            except Exception as e:
                                logger.error(f"Error adding follow-up response to chat history: {e}")
                                # Try to ensure at least something is shown
                                # Find tool response content
                                tool_results = []
                                for msg in st.session_state.tool_chat_history:
                                    if msg.get("type") == "tool_response":
                                        tool_results.append(msg.get("content", ""))
                                
                                if tool_results:
                                    latest_result = tool_results[-1]
                                    # Only add a summary if we haven't already shown the result
                                    summary_shown = False
                                    for msg in st.session_state.tool_chat_history[-3:]:
                                        if msg.get("role") == "assistant" and latest_result in msg.get("content", ""):
                                            summary_shown = True
                                            break
                                    
                                    if not summary_shown:
                                        st.session_state.add_to_chat_history({
                                            "role": "assistant",
                                            "content": f"The calculation result is: {latest_result}"
                                        })
                        
                        # If no tool calls were made but we detected JSON in the response, try to use the JSON tool
                        if should_use_json_tool and json_text and "json_tool" in TOOL_IMPLEMENTATIONS and "tool_calls" not in assistant_message:
                            logger.info(f"Manually invoking JSON tool with extracted JSON: {json_text[:100]}...")
                            try:
                                # Create a synthetic tool call
                                import uuid
                                tool_call_id = str(uuid.uuid4())
                                
                                # Add a note in the assistant message to inform the user what's happening
                                assistant_history_msg["content"] += "\n\n*Note: Using JSON tool to process this data*"
                                
                                # Add the assistant message to chat history first with duplicate prevention
                                st.session_state.add_to_chat_history(assistant_history_msg)
                                
                                # Pre-process JSON to fix common issues with single quotes and escaping
                                # This duplicates some of the logic in the JSON tool itself but ensures we catch issues
                                # before passing to the tool
                                processed_json = json_text
                                
                                # Replace single quotes with double quotes (accounting for nested quotes)
                                if "'" in processed_json:
                                    try:
                                        # Handle cases with single quoted values
                                        import re
                                        # Look for places where we have single-quoted strings
                                        quoted_pattern = r"'([^']*?)'"
                                        
                                        # Replace with double quotes, but ensure nested quotes are handled
                                        def replacer(match):
                                            # Get the content of the single-quoted string
                                            content = match.group(1)
                                            # Replace any double quotes in the content with escaped double quotes
                                            content = content.replace('"', '\\"')
                                            # Return the content wrapped in double quotes
                                            return f'"{content}"'
                                        
                                        # Apply the replacement
                                        processed_json = re.sub(quoted_pattern, replacer, processed_json)
                                        logger.info(f"Fixed JSON single quotes: {processed_json[:100]}...")
                                    except Exception as re_error:
                                        logger.error(f"Error fixing JSON quotes: {re_error}")
                                
                                # Process with JSON tool using the preprocessed text
                                result = TOOL_IMPLEMENTATIONS["json_tool"](text=processed_json, operation="repair")
                                
                                # Add tool response to chat history with duplicate prevention
                                tool_response_msg = {
                                    "role": "assistant",
                                    "type": "tool_response",
                                    "content": result,
                                    "tool_call_id": tool_call_id
                                }
                                
                                # Track if we added a message so we know whether to add the follow-up
                                response_added = st.session_state.add_to_chat_history(tool_response_msg)
                                
                                if response_added:
                                    # Force an immediate UI update to show the JSON tool response
                                    st.rerun()
                                
                                    # Add a follow-up message explaining what happened
                                    st.session_state.add_to_chat_history({
                                        "role": "assistant",
                                        "content": "I've processed the JSON data using the JSON tool to properly format and validate it."
                                    })
                            except Exception as e:
                                logger.error(f"Error in manual JSON tool invocation: {e}")
                                # Make sure the original message is added if not already done
                                st.session_state.add_to_chat_history(assistant_history_msg)
                        else:
                            # Add the assistant message to the chat history if not already added
                            st.session_state.add_to_chat_history(assistant_history_msg)
                        
                        # Rerun to update the UI
                        st.rerun()
                    else:
                        message_placeholder.error("Received an unexpected response format from the model.")
                
                except Exception as e:
                    error_message = str(e)
                    
                    # Check for common tool-related error messages
                    if error_message == "Error: 'id'":
                        # This is an error related to missing 'id' field in tool calls
                        friendly_error = "Error: Tool response is missing required 'id' field"
                        recommendations = """
                        The model returned a tool call without the required 'id' field.
                        This can happen with some models that don't fully implement the tools format.
                        
                        Try using a different model like:
                        - llama3 (best support for tools)
                        - mistral
                        - qwen2
                        
                        You can pull one of these models using: `ollama pull llama3`
                        """
                        
                        message_placeholder.error(friendly_error)
                        message_placeholder.info(recommendations)
                        
                        # Log the error for diagnostics
                        logger.error(f"Missing 'id' field in tool call with model '{st.session_state.selected_tool_model}'")
                        
                        # Add error message to chat history
                        st.session_state.tool_chat_history.append({
                            "role": "assistant",
                            "content": f"{friendly_error}\n\n{recommendations}"
                        })
                        
                        # Rerun to update the UI
                        st.rerun()
                        
                    elif "does not support tools" in error_message.lower() or "function calling" in error_message.lower() or "status code: 400" in error_message.lower():
                        # This is a tool support error
                        # Extract the actual model name from the error message if possible
                        actual_model = error_message.split("'")[1] if "'" in error_message else st.session_state.selected_tool_model
                        friendly_error = f"Error: Model '{actual_model}' does not support tool/function calling."
                        
                        # Provide more helpful instructions
                        recommendations = """
                        To use tools/function calling with Ollama, you need a model that supports this capability.
                        
                        Recommended models:
                        - llama3 (best support for tools)
                        - mistral
                        - qwen
                        - phi3
                        
                        You can pull one of these models using: `ollama pull llama3`
                        
                        Learn more: https://ollama.com/search?c=tools
                        """
                        
                        message_placeholder.error(friendly_error)
                        message_placeholder.info(recommendations)
                        
                        # Log the error for diagnostics
                        logger.error(f"Tool support error with model '{st.session_state.selected_tool_model}': {error_message}")
                        
                        # Add error message to chat history
                        st.session_state.tool_chat_history.append({
                            "role": "assistant",
                            "content": f"{friendly_error}\n\n{recommendations}"
                        })
                    # Check if we already have any successful tool responses
                    elif any(msg.get("type") == "tool_response" for msg in st.session_state.tool_chat_history[-5:]):
                        # We already have a result, so this error is likely just a follow-up issue
                        # Just log it without showing to the user
                        logger.warning(f"Error occurred after successful result display: {error_message}")
                        # Don't add error message to chat history
                        return
                        
                    elif "object of type toolcall" in error_message.lower() and "json serializable" in error_message.lower():
                        # This is a JSON serialization error with ToolCall objects
                        friendly_error = "Error: The model returned a response format that couldn't be properly processed."
                        recommendations = """
                        There was an issue with the tool call format returned by the model.
                        
                        This can happen with:
                        1. Older versions of the Ollama package
                        2. Models that don't fully implement the tool calling format
                        
                        Try:
                        - Updating Ollama to the latest version
                        - Using a different model like llama3, which has better tools support
                        - Simplifying your query
                        """
                        
                        message_placeholder.error(friendly_error)
                        message_placeholder.info(recommendations)
                        
                        # Log the error for diagnostics
                        logger.error(f"JSON serialization error with model '{st.session_state.selected_tool_model}': {error_message}")
                        
                        # Add error message to chat history
                        st.session_state.tool_chat_history.append({
                            "role": "assistant",
                            "content": f"{friendly_error}\n\n{recommendations}"
                        })
                        
                    elif "schema" in error_message.lower() and "validation" in error_message.lower():
                        # This is likely a tool schema validation error
                        friendly_error = f"Error: Tool schema validation failed. There may be an issue with your tool definition. Details: {error_message}"
                        message_placeholder.error(friendly_error)
                        
                        # Add error message to chat history
                        st.session_state.tool_chat_history.append({
                            "role": "assistant",
                            "content": friendly_error
                        })
                    else:
                        # Generic error handling
                        formatted_error = f"Error: {error_message}"
                        message_placeholder.error(formatted_error)
                        
                        # Add error message to chat history
                        st.session_state.tool_chat_history.append({
                            "role": "assistant",
                            "content": formatted_error
                        })
                    
                    # Rerun to update the UI
                    st.rerun()

if __name__ == "__main__":
    tool_playground()