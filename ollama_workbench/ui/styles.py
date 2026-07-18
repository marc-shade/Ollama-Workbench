"""
Centralized styling for Ollama Workbench.

Since Streamlit 1.47 nearly all visual styling (colors, fonts, radii,
borders, sidebar palette) is defined declaratively in
`.streamlit/config.toml`. This module only:

1. exposes the COLORS palette that non-theme components consume
   (streamlit_option_menu navigation styling in main.py), kept in sync
   with the config.toml theme, and
2. injects the few layout tweaks the theme system cannot express
   (content max-width, transparent header, option-menu item spacing).

The old 400-line CSS payload targeted a legacy custom-HTML chat UI
(.message, .chat-container, .model-selector) that no longer exists in
the rendered DOM, and its `font-family !important` override defeated
theme fonts. Do not reintroduce broad `!important` element overrides
here - put visual changes in config.toml instead.
"""

import streamlit as st

# Palette for components that cannot read the Streamlit theme directly
# (e.g. streamlit_option_menu). Keep in sync with .streamlit/config.toml.
COLORS = {
    "light": {
        "primary": "#E85A1A",
        "secondary": "#C2410C",
        "accent": "#FF8A50",
        "background": "#FFFFFF",
        "panel": "#F7F8FA",
        "sidebar": "#F2F4F7",
        "card": "#FFFFFF",
        "text": "#1A202C",
        "text_secondary": "#5A6472",
        "border": "#E3E7EC",
        "hover": "#EDF0F3",
        "selected": "#E85A1A1A",
        "selected_text": "#C2410C",
    },
    "dark": {
        "primary": "#E85A1A",
        "secondary": "#FF8A50",
        "accent": "#FF8A50",
        "background": "#101418",
        "panel": "#1A2027",
        "sidebar": "#0B0F13",
        "card": "#1A2027",
        "text": "#E8EAED",
        "text_secondary": "#98A2B0",
        "border": "#2A323C",
        "hover": "#232B34",
        "selected": "#E85A1A26",
        "selected_text": "#FF8A50",
    },
}


def detect_theme():
    """Use Streamlit's built-in theme detection."""
    return "light" if st.get_option("theme.base") == "light" else "dark"


def get_theme_colors(theme):
    """Get the theme colors for the current theme."""
    return COLORS.get(theme, COLORS["dark"])


def apply_styles():
    """Inject the layout tweaks the theme system cannot express.

    Returns (colors, theme) for callers that style non-theme components.
    """
    theme = detect_theme()
    colors = get_theme_colors(theme)

    st.markdown(
        """
        <style>
        /* Keep the toolbar visually quiet */
        .stApp > header {
            background-color: transparent;
        }

        /* Comfortable content column */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1000px;
        }

        /* streamlit_option_menu items (rendered in a component iframe-free
           nav; the theme system does not reach these classes) */
        .nav-link {
            display: flex;
            align-items: center;
            padding: 0.5rem 0.75rem;
            margin: 0.25rem 0;
            border-radius: 0.5rem;
            transition: background-color 0.15s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    return colors, theme
