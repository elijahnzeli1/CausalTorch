# Contributing to CausalTorch

Thank you for your interest in contributing to CausalTorch! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear title and description
- Steps to reproduce the bug
- Expected and actual behavior
- Your environment information (OS, Python version, PyTorch version)

### Suggesting Enhancements

Feature requests are welcome! Please create an issue with:
- A clear title and description
- Explanation of why this feature would be useful
- Potential implementation approach (if you have ideas)

### Pull Requests

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for your changes
5. Run the tests: `pytest`
6. Commit your changes: `git commit -m 'Add feature'`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/elijahnzeli1/CausalTorch.git
cd CausalTorch

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Code Style

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.

- Use 4 spaces for indentation
- Use docstrings for all public classes and functions
- Run `black` and `isort` before committing

## Testing

Please add tests for new features. We use PyTest for testing.

## Documentation

Update documentation when changing functionality:
1. Update docstrings
2. Update README.md if needed
3. Add examples if appropriate

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.