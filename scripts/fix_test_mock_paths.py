#!/usr/bin/env python3
"""Fix mock patch paths in test files after package reorganization.

Replaces old flat module names with new package paths in patch() calls.
"""
import re
import glob

# Mapping of old module names to new package paths
MOCK_PATH_MAP = {
    "brainstorm.": "ollama_workbench.workflows.brainstorm.",
    "projects.": "ollama_workbench.workflows.projects.",
    "build.": "ollama_workbench.workflows.build.",
    "build_manager.": "ollama_workbench.workflows.build.",
    "research.": "ollama_workbench.workflows.research.",
    "nodes.": "ollama_workbench.workflows.nodes.",
    "chat_interface.": "ollama_workbench.chat.chat_interface.",
    "multimodel_chat.": "ollama_workbench.chat.multimodel_chat.",
    "voice_interface.": "ollama_workbench.chat.voice_interface.",
    "voice_utils.": "ollama_workbench.chat.voice_utils.",
    "ollama_utils.": "ollama_workbench.providers.ollama_utils.",
    "openai_utils.": "ollama_workbench.providers.openai_utils.",
    "groq_utils.": "ollama_workbench.providers.groq_utils.",
    "mistral_utils.": "ollama_workbench.providers.mistral_utils.",
    "external_providers.": "ollama_workbench.providers.external_providers.",
    "server_configuration.": "ollama_workbench.server.server_configuration.",
    "server_monitoring.": "ollama_workbench.server.server_monitoring.",
    "performance_metrics.": "ollama_workbench.server.performance_metrics.",
    "file_management.": "ollama_workbench.ui.file_management.",
    "prompts.": "ollama_workbench.ui.prompts.",
    "structured_output.": "ollama_workbench.ui.structured_output.",
    "repo_docs.": "ollama_workbench.knowledge.repo_docs.",
    "web_to_corpus.": "ollama_workbench.knowledge.web_to_corpus.",
    "error_handling.": "ollama_workbench.core.error_handling.",
    "collaborative_workspace.": "ollama_workbench.chat.collaborative_workspace.",
    "enhanced_corpus.": "ollama_workbench.knowledge.enhanced_corpus.",
    "corpus_management.": "ollama_workbench.knowledge.corpus_management.",
    "simplified_rag.": "ollama_workbench.knowledge.simplified_rag.",
}

# These should NOT be remapped (stdlib, third-party, or already correct)
SKIP_PREFIXES = [
    "ollama_workbench.",  # Already correct
    "streamlit.",
    "builtins.",
    "os.",
    "requests.",
    "subprocess.",
    "json.",
    "time.",
    "unittest.",
    "pytest.",
    "threading.",
    "logging.",
    "datetime.",
    "tts_server.",  # Separate package, stays as-is
    "persona_lab.",  # Separate package
    "observability.",  # Separate package
]

def should_skip(old_path):
    """Check if this path should not be remapped."""
    for prefix in SKIP_PREFIXES:
        if old_path.startswith(prefix):
            return True
    return False

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    changes = 0

    # Find all patch('module.xxx') patterns
    # Match patch('old_module.something') and @patch('old_module.something')
    for old_prefix, new_prefix in MOCK_PATH_MAP.items():
        # Only replace within patch() calls - look for patch(' or patch("
        for quote in ["'", '"']:
            old_pattern = f"patch({quote}{old_prefix}"
            new_pattern = f"patch({quote}{new_prefix}"
            if old_pattern in content:
                count = content.count(old_pattern)
                content = content.replace(old_pattern, new_pattern)
                changes += count

    if changes > 0:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  {filepath}: {changes} replacements")

    return changes

def main():
    total = 0
    test_files = sorted(glob.glob("tests/test_*.py"))

    for filepath in test_files:
        count = fix_file(filepath)
        total += count

    print(f"\nTotal: {total} mock path replacements across {len(test_files)} files")

if __name__ == "__main__":
    main()
