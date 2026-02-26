#!/usr/bin/env python3
"""Test that all modules in the ollama_workbench package can be imported."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/claude/matplotlib')

import importlib
import warnings
warnings.filterwarnings('ignore')

errors = []
ok_count = 0

for root, dirs, files in os.walk('ollama_workbench'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in sorted(files):
        if f.endswith('.py') and f != '__init__.py':
            mod_path = os.path.join(root, f).replace('/', '.').replace('.py', '')
            try:
                importlib.import_module(mod_path)
                ok_count += 1
            except Exception as e:
                err_msg = str(e).split('\n')[0][:120]
                errors.append((mod_path, err_msg))

if errors:
    print(f'{len(errors)} modules had import errors (out of {ok_count + len(errors)}):')
    for m, e in errors:
        print(f'  {m}: {e}')
    sys.exit(1)
else:
    print(f'All {ok_count} modules imported successfully')
