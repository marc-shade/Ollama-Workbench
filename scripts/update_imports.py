#!/usr/bin/env python3
"""Mechanically update all imports after file reorganization into ollama_workbench/ package."""

import re
import os
import sys

# Mapping: old module name -> new dotted path
MODULE_MAP = {
    # core/
    'config': 'ollama_workbench.core.config',
    'session_utils': 'ollama_workbench.core.session_utils',
    'db_init': 'ollama_workbench.core.db_init',
    'error_handling': 'ollama_workbench.core.error_handling',

    # providers/
    'ollama_utils': 'ollama_workbench.providers.ollama_utils',
    'openai_utils': 'ollama_workbench.providers.openai_utils',
    'groq_utils': 'ollama_workbench.providers.groq_utils',
    'mistral_utils': 'ollama_workbench.providers.mistral_utils',
    'external_providers': 'ollama_workbench.providers.external_providers',

    # chat/
    'chat_interface': 'ollama_workbench.chat.chat_interface',
    'enhanced_chat_interface': 'ollama_workbench.chat.enhanced_chat_interface',
    'multimodel_chat': 'ollama_workbench.chat.multimodel_chat',
    'multimodal_chat': 'ollama_workbench.chat.multimodal_chat',
    'voice_interface': 'ollama_workbench.chat.voice_interface',
    'persona_chat': 'ollama_workbench.chat.persona_chat',
    'collaborative_workspace': 'ollama_workbench.chat.collaborative_workspace',
    'canvas': 'ollama_workbench.chat.canvas',
    'voice_utils': 'ollama_workbench.chat.voice_utils',
    'tts_utils': 'ollama_workbench.chat.tts_utils',

    # workflows/
    'build': 'ollama_workbench.workflows.build',
    'research': 'ollama_workbench.workflows.research',
    'brainstorm': 'ollama_workbench.workflows.brainstorm',
    'projects': 'ollama_workbench.workflows.projects',
    'nodes': 'ollama_workbench.workflows.nodes',
    'agents': 'ollama_workbench.workflows.agents',
    'info_brainstorm': 'ollama_workbench.workflows.info_brainstorm',

    # knowledge/
    'simplified_rag': 'ollama_workbench.knowledge.simplified_rag',
    'enhanced_corpus': 'ollama_workbench.knowledge.enhanced_corpus',
    'corpus_management': 'ollama_workbench.knowledge.corpus_management',
    'repo_docs': 'ollama_workbench.knowledge.repo_docs',
    'web_to_corpus': 'ollama_workbench.knowledge.web_to_corpus',
    'search_libraries': 'ollama_workbench.knowledge.search_libraries',
    'chroma_client': 'ollama_workbench.knowledge.chroma_client',

    # models/
    'model_comparison': 'ollama_workbench.models.model_comparison',
    'model_tests': 'ollama_workbench.models.model_tests',
    'feature_test': 'ollama_workbench.models.feature_test',
    'vision_comparison': 'ollama_workbench.models.vision_comparison',
    'local_models': 'ollama_workbench.models.local_models',
    'pull_model': 'ollama_workbench.models.pull_model',
    'show_model': 'ollama_workbench.models.show_model',
    'remove_model': 'ollama_workbench.models.remove_model',
    'update_models': 'ollama_workbench.models.update_models',
    'model_management': 'ollama_workbench.models.model_management',
    'model_capabilities': 'ollama_workbench.models.model_capabilities',
    'model_capability_registry': 'ollama_workbench.models.model_capability_registry',
    'model_onboarding': 'ollama_workbench.models.model_onboarding',
    'test_visualization': 'ollama_workbench.models.test_visualization',

    # server/
    'server_configuration': 'ollama_workbench.server.server_configuration',
    'server_monitoring': 'ollama_workbench.server.server_monitoring',
    'performance_metrics': 'ollama_workbench.server.performance_metrics',
    'openai_compatibility': 'ollama_workbench.server.openai_compatibility',

    # ui/
    'styles': 'ollama_workbench.ui.styles',
    'prompts': 'ollama_workbench.ui.prompts',
    'file_management': 'ollama_workbench.ui.file_management',
    'structured_output': 'ollama_workbench.ui.structured_output',
    'tool_playground': 'ollama_workbench.ui.tool_playground',
    'contextual_response': 'ollama_workbench.ui.contextual_response',
    'welcome': 'ollama_workbench.ui.welcome',
    'global_vrm_loader': 'ollama_workbench.ui.global_vrm_loader',
}

# Directories NOT in our package (leave imports alone):
SKIP_DIRS = {'venv', '.venv', '__pycache__', 'node_modules', '.git', 'backup_files', 'tmp', 'data', 'extension'}

def should_process(filepath):
    """Check if file should be processed."""
    parts = filepath.split(os.sep)
    for skip in SKIP_DIRS:
        if skip in parts:
            return False
    return filepath.endswith('.py')


def transform_line(line, filepath=''):
    """Transform a single import line."""
    original = line

    # Pattern 1: "from MODULE import ..." (including wildcard)
    m = re.match(r'^(\s*)(from\s+)(\w+)(\s+import\s+.*)$', line)
    if m:
        indent, from_kw, module, rest = m.groups()
        if module in MODULE_MAP:
            new_module = MODULE_MAP[module]
            # Check if this file is IN the same sub-package -> use relative import
            if filepath:
                file_package = get_file_package(filepath)
                target_package = '.'.join(new_module.split('.')[:-1])
                if file_package and file_package == target_package:
                    # Same sub-package: use relative import
                    target_module = new_module.split('.')[-1]
                    return f"{indent}{from_kw}.{target_module}{rest}\n"
            return f"{indent}{from_kw}{new_module}{rest}\n"

    # Pattern 2: "import MODULE" (bare import)
    m = re.match(r'^(\s*)(import\s+)(\w+)(\s*(?:#.*)?)$', line)
    if m:
        indent, import_kw, module, rest = m.groups()
        if module in MODULE_MAP:
            new_module = MODULE_MAP[module]
            return f"{indent}{import_kw}{new_module} as {module}{rest}\n"

    # Pattern 3: "import MODULE as ALIAS"
    m = re.match(r'^(\s*)(import\s+)(\w+)(\s+as\s+\w+\s*(?:#.*)?)$', line)
    if m:
        indent, import_kw, module, rest = m.groups()
        if module in MODULE_MAP:
            new_module = MODULE_MAP[module]
            return f"{indent}{import_kw}{new_module}{rest}\n"

    return line


def get_file_package(filepath):
    """Get the package path for a file within ollama_workbench/."""
    # Normalize path
    norm = os.path.normpath(filepath)
    parts = norm.split(os.sep)
    # Find ollama_workbench in path
    if 'ollama_workbench' in parts:
        idx = parts.index('ollama_workbench')
        # Get sub-package (e.g., ollama_workbench.providers)
        pkg_parts = parts[idx:-1]  # exclude filename
        return '.'.join(pkg_parts)
    return None


def process_file(filepath, dry_run=False):
    """Process a single file, updating imports."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return 0

    changes = 0
    new_lines = []
    for i, line in enumerate(lines):
        new_line = transform_line(line, filepath)
        if new_line != line:
            changes += 1
            if dry_run:
                print(f"  L{i+1}: {line.rstrip()} -> {new_line.rstrip()}")
        new_lines.append(new_line)

    if changes > 0 and not dry_run:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return changes


def main():
    dry_run = '--dry-run' in sys.argv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dry_run:
        print("=== DRY RUN (no files modified) ===\n")

    total_changes = 0
    files_changed = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip unwanted directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if not should_process(filepath):
                continue

            changes = process_file(filepath, dry_run)
            if changes > 0:
                rel = os.path.relpath(filepath, root)
                print(f"{'Would update' if dry_run else 'Updated'} {rel}: {changes} import(s)")
                total_changes += changes
                files_changed += 1

    print(f"\n{'Would update' if dry_run else 'Updated'} {total_changes} imports across {files_changed} files")


if __name__ == '__main__':
    main()
