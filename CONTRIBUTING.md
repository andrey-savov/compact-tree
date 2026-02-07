# Contributing to compact-tree

Thank you for your interest in contributing to compact-tree! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/andrey-savov/compact-tree.git
   cd compact-tree
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Workflow

### Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure they follow the project's coding style

3. **Write or update tests** to cover your changes in `test_compact_tree.py`

4. **Run the test suite** to ensure everything passes:
   ```bash
   pytest
   ```

5. **Run tests with coverage** (optional):
   ```bash
   pytest --cov=compact_tree --cov-report=html
   ```

### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep lines under 88 characters when practical
- Use type hints where appropriate

### Commit Messages

- Write clear, concise commit messages
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issues and pull requests when relevant

Example:
```
Add support for nested list values

- Extend CompactTree to handle list values
- Add tests for list serialization
- Update documentation

Fixes #123
```

## Testing

All contributions should include tests:

- **Unit tests** for new functions or methods
- **Integration tests** for new features
- **Regression tests** for bug fixes

Run tests with:
```bash
pytest test_compact_tree.py -v
```

## Submitting Changes

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Describe what changes you made and why
   - Include any breaking changes or migration notes

3. **Wait for review**:
   - Address any feedback from reviewers
   - Make requested changes in new commits
   - Push updates to your branch

## Reporting Bugs

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal code example that demonstrates the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, package versions

## Suggesting Features

Feature requests are welcome! Please:

- **Check existing issues** to avoid duplicates
- **Describe the use case** and why the feature would be useful
- **Provide examples** of how it would work
- **Consider implementation** if you have ideas

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the "question" label
- Start a discussion on GitHub Discussions (if enabled)

## Code of Conduct

Please be respectful and professional in all interactions. We aim to foster an inclusive and welcoming community.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
