# Building and Publishing compact-tree

This guide explains how to build and publish the `compact-tree` package.

## Prerequisites

```bash
pip install build twine
```

## Building the Package

### Option 1: Using Python build module (recommended)

```bash
python -m build
```

This creates both source distribution (.tar.gz) and wheel (.whl) in the `dist/` directory.

### Option 2: Using Makefile (Unix/Linux/Mac)

```bash
make build
```

### Option 3: Using PowerShell (Windows)

```powershell
# Clean previous builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Build the package
python -m build
```

## Installing Locally for Development

Install in editable mode (changes to source code take effect immediately):

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Testing the Built Package

After building, you can install the wheel locally to test:

```bash
pip install dist/savov_compact_tree-1.1.0-py3-none-any.whl
```

Or test in a fresh virtual environment:

```bash
# Create new environment
python -m venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Install the built package
pip install dist/savov_compact_tree-1.1.0-py3-none-any.whl

# Test it
python -c "from compact_tree import CompactTree; print('✅ Import successful!')"
```

## Publishing to PyPI

### Test PyPI (recommended first)

1. Create account at https://test.pypi.org/
2. Upload to Test PyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

3. Test installation from Test PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ savov-compact-tree
```

### Production PyPI

1. Create account at https://pypi.org/
2. Upload to PyPI:

```bash
python -m twine upload dist/*
```

3. Install from PyPI:

```bash
pip install compact-tree
```

## Using API Tokens (Recommended)

Instead of username/password, use API tokens:

1. Generate token at https://pypi.org/manage/account/token/
2. Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
username = __token__
password = pypi-your-test-api-token-here
```

## Automating Releases with GitHub Actions

The project includes a workflow template. Create `.github/workflows/publish.yml`:

```yaml
name: Upload to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Add your PyPI token to GitHub repository secrets as `PYPI_API_TOKEN`.

## Version Management

Update version in `pyproject.toml` before building:

```toml
[project]
version = "1.1.0"  # Increment for new release
```

Follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

## Clean Build Artifacts

```bash
make clean
# or manually:
rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .pytest_cache/ __pycache__/
```
