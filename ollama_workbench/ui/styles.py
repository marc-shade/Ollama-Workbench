"""
Centralized styling for Ollama Workbench that follows the Open WebUI design approach.

This file contains all CSS styles needed to create a modern, consistent UI experience
throughout the application. It provides both light and dark theme styles.
"""

import streamlit as st
from streamlit_javascript import st_javascript

# Common colors for the application
COLORS = {
    # Light mode colors
    "light": {
        "primary": "#2563EB",  # Blue primary color
        "secondary": "#4F46E5", # Indigo secondary color
        "accent": "#7C3AED",    # Purple accent
        "background": "#FFFFFF",
        "panel": "#F9FAFB",
        "sidebar": "#F3F4F6", 
        "card": "#FFFFFF",
        "text": "#111827",
        "text_secondary": "#4B5563",
        "border": "#E5E7EB",
        "hover": "#F3F4F6",
        "selected": "#EFF6FF",
        "selected_text": "#2563EB"
    },
    # Dark mode colors
    "dark": {
        "primary": "#3B82F6",  # Blue primary color
        "secondary": "#6366F1", # Indigo secondary color
        "accent": "#8B5CF6",    # Purple accent
        "background": "#111827", # Dark slate
        "panel": "#1F2937",
        "sidebar": "#1F2937", 
        "card": "#1F2937",
        "text": "#F9FAFB",
        "text_secondary": "#9CA3AF",
        "border": "#374151",
        "hover": "#374151",
        "selected": "#2563EB30",
        "selected_text": "#60A5FA"
    }
}

def detect_theme():
    """Use Streamlit's built-in theme detection."""
    # Let Streamlit handle theme detection
    return "light" if st.get_option("theme.base") == "light" else "dark"

def get_theme_colors(theme):
    """Get the theme colors for the current theme."""
    return COLORS.get(theme, COLORS["light"])

def apply_base_styles():
    """Apply base styles that apply to both light and dark mode."""
    st.markdown("""
        <style>
        /* Base Typography */
        body, h1, h2, h3, h4, h5, h6, p, li, div {
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 
                         "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
            font-weight: 400;
        }
        
        /* Remove borders and padding from all containers */
        .stApp > header {
            background-color: transparent !important;
            border: none !important;
        }
        
        div.stButton > button {
            border-radius: 0.375rem;
            font-weight: 500;
            transition: all 0.15s ease-in-out;
        }
        
        /* Streamlit containers */
        [data-testid="stVerticalBlock"] {
            gap: 0.75rem !important;
        }
        
        /* Chat interface elements */
        .stChatMessage {
            border-radius: 0.5rem !important;
            padding: 0.75rem 1rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        .stChatInputContainer {
            border-radius: 0.5rem !important;
            padding: 0.5rem !important;
        }
        
        /* Adjustments for mobile */
        @media (max-width: 768px) {
            .stApp {
                padding: 0.5rem !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def apply_functional_styles(colors):
    """Apply only functional UI styles that enhance the experience without overriding Streamlit's theming."""
    
    st.markdown(f"""
        <style>
        /* Main container adjustments */
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1000px;
        }}
        
        /* Navigation enhancements */
        .nav-link {{
            display: flex !important;
            align-items: center !important;
            padding: 0.5rem 0.75rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 0.375rem !important;
            transition: all 0.15s ease !important;
        }}
        
        /* Model selector UI */
        .model-selector {{
            position: fixed !important;
            top: 1rem !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            z-index: 1000 !important;
            border-radius: 0.5rem !important;
            padding: 0.5rem 0.75rem !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }}
        
        /* Chat layout improvements */
        .chat-container {{
            display: flex !important;
            flex-direction: column !important;
            height: calc(100vh - 5rem) !important;
            max-height: calc(100vh - 5rem) !important;
            overflow: hidden !important;
        }}
        
        /* Message history improvements */
        .message-history {{
            flex: 1 !important;
            overflow-y: auto !important;
            padding: 1rem !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 1rem !important;
        }}
        
        /* Message bubbles */
        .user-message-container {{
            display: flex !important;
            justify-content: flex-end !important;
            margin-bottom: 1rem !important;
        }}
        
        .assistant-message-container {{
            display: flex !important;
            justify-content: flex-start !important;
            margin-bottom: 1rem !important;
        }}
        
        /* Input area enhancements */
        .input-area {{
            padding: 0.75rem !important;
            border-top-width: 1px !important;
            border-top-style: solid !important;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    return colors

def sidebar_styles(colors):
    """Apply specific styles to the sidebar navigation."""
    st.markdown(f"""
        <style>
        .sidebar-nav {{
            margin-top: 1rem !important;
        }}
        
        .sidebar-section {{
            margin-bottom: 1.5rem !important;
        }}
        
        .sidebar-section-title {{
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            color: {colors['text_secondary']} !important;
            margin-bottom: 0.5rem !important;
            padding-left: 0.75rem !important;
        }}
        
        /* Hide default Streamlit sidebar elements */
        section[data-testid="stSidebar"] div.stButton button {{
            background-color: transparent !important;
            color: {colors['text']} !important;
            text-align: left !important;
            font-weight: 400 !important;
            padding: 0.5rem 0.75rem !important;
            border-radius: 0.375rem !important;
            margin-bottom: 0.25rem !important;
        }}
        
        section[data-testid="stSidebar"] div.stButton button:hover {{
            background-color: {colors['hover']} !important;
        }}
        
        /* Custom scrollbar for sidebar */
        section[data-testid="stSidebar"] {{
            scrollbar-width: thin;
            scrollbar-color: {colors['border']} transparent;
        }}
        
        section[data-testid="stSidebar"]::-webkit-scrollbar {{
            width: 6px;
        }}
        
        section[data-testid="stSidebar"]::-webkit-scrollbar-track {{
            background: transparent;
        }}
        
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {{
            background-color: {colors['border']};
            border-radius: 20px;
        }}
        </style>
    """, unsafe_allow_html=True)

def chat_interface_styles(colors):
    """Apply specific styles to the chat interface."""
    st.markdown(f"""
        <style>
        /* Chat container */
        .chat-container {{
            display: flex !important;
            flex-direction: column !important;
            height: calc(100vh - 5rem) !important;
            max-height: calc(100vh - 5rem) !important;
            overflow: hidden !important;
        }}
        
        /* Message history container */
        .message-history {{
            flex: 1 !important;
            overflow-y: auto !important;
            padding: 1rem !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 1rem !important;
        }}
        
        /* Message bubbles */
        .message {{
            display: flex !important;
            margin-bottom: 1rem !important;
            max-width: 80% !important;
        }}
        
        .message.user {{
            align-self: flex-end !important;
        }}
        
        .message.assistant {{
            align-self: flex-start !important;
        }}
        
        .message-content {{
            padding: 0.75rem 1rem !important;
            border-radius: 0.75rem !important;
            font-size: 0.9375rem !important;
            line-height: 1.5 !important;
        }}
        
        .message.user .message-content {{
            background-color: {colors['primary']} !important;
            color: white !important;
            border-radius: 0.75rem 0.75rem 0 0.75rem !important;
        }}
        
        .message.assistant .message-content {{
            background-color: {colors['panel']} !important;
            color: {colors['text']} !important;
            border-radius: 0 0.75rem 0.75rem 0.75rem !important;
        }}
        
        /* Input area */
        .input-area {{
            padding: 1rem !important;
            border-top: 1px solid {colors['border']} !important;
            background-color: {colors['background']} !important;
        }}
        
        .input-container {{
            position: relative !important;
            display: flex !important;
            align-items: center !important;
        }}
        
        .chat-input {{
            flex: 1 !important;
            padding: 0.75rem 1rem !important;
            padding-right: 3rem !important;
            border-radius: 0.5rem !important;
            border: 1px solid {colors['border']} !important;
            background-color: {colors['card']} !important;
            color: {colors['text']} !important;
            font-size: 0.9375rem !important;
            resize: none !important;
            outline: none !important;
            transition: border-color 0.15s ease-in-out !important;
        }}
        
        .chat-input:focus {{
            border-color: {colors['primary']} !important;
            box-shadow: 0 0 0 3px {colors['primary']}33 !important;
        }}
        
        .send-button {{
            position: absolute !important;
            right: 0.5rem !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            background-color: {colors['primary']} !important;
            color: white !important;
            border: none !important;
            border-radius: 0.375rem !important;
            padding: 0.375rem 0.75rem !important;
            cursor: pointer !important;
            transition: background-color 0.15s ease-in-out !important;
        }}
        
        .send-button:hover {{
            background-color: {colors['secondary']} !important;
        }}
        
        /* Floating model selector */
        .model-selector {{
            position: fixed !important;
            top: 1rem !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            z-index: 1000 !important;
            background-color: {colors['card']} !important;
            border: 1px solid {colors['border']} !important;
            border-radius: 0.5rem !important;
            padding: 0.5rem 0.75rem !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }}
        
        .model-selector select {{
            background-color: transparent !important;
            border: none !important;
            color: {colors['text']} !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            padding: 0.25rem 0.5rem !important;
            outline: none !important;
        }}
        
        /* Loading indicator */
        .loading-indicator {{
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
            color: {colors['text_secondary']} !important;
            font-size: 0.875rem !important;
            padding: 0.5rem !important;
        }}
        
        /* System message */
        .system-message {{
            color: {colors['text_secondary']} !important;
            font-size: 0.875rem !important;
            text-align: center !important;
            padding: 0.5rem !important;
            margin: 0.5rem 0 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

def apply_styles():
    """Apply core styles to the application without overriding Streamlit's theming."""
    # Detect theme using Streamlit's built-in theme
    theme = detect_theme()
    
    # Get colors for current theme
    colors = get_theme_colors(theme)
    
    # Apply only essential styles that don't interfere with Streamlit's theming
    apply_base_styles()
    
    # Apply only functional UI elements that enhance the experience
    apply_functional_styles(colors)
    
    return colors, theme