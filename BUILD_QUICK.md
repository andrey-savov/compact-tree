# Quick Build & Test Guide

## Build the package

```powershell
# Using the virtual environment Python
.\.venv\Scripts\python.exe -m build
```

## Test the built package locally

```powershell
# Install in a test environment
pip install dist\savov_compact_tree-1.0.0-py3-none-any.whl

# Or install in editable mode for development
pip install -e .
```

## Quick test

```powershell
python -c "from compact_tree import CompactTree; print('✅ Import successful!')"
```

## Run tests

```powershell
pytest test_compact_tree.py test_marisa_trie.py -v
```

See [BUILD.md](BUILD.md) for complete publishing instructions.
