# Contributing to AGILE

Thank you for your interest in contributing to the AGILE project! This document provides guidelines and instructions for contributing.

## Development Environment Setup

Please refer to the [README.md](README.md) file for detailed instructions on setting up the development environment.

## Code Style and Standards

This project follows these coding standards:

- **Python**: We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications as specified in our `.flake8` configuration.
- **Formatting**: We use Black for code formatting and isort for import sorting.
- **Type Hints**: We encourage the use of type hints for better code readability and maintainability.

Our pre-commit hooks will automatically check and enforce these standards.

## Pull Request Process

1. Fork the repository and create a new branch from `main` for your feature or bugfix.
2. Make your changes, ensuring they follow our code style guidelines.
3. Add tests for your changes if applicable.
4. Update documentation as necessary.
5. Run the pre-commit hooks on your changes:
   ```bash
   pre-commit run --all-files
   ```
6. Submit a pull request with a clear description of the changes and any relevant issue numbers.

## Commit Message Guidelines

We follow a simplified version of the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes that don't affect code behavior
- `refactor:` for code refactoring without behavior changes
- `test:` for adding or modifying tests

Example: `feat: add new locomotion controller`

## Testing

Please ensure that your code changes include appropriate tests. Run existing tests to make sure your changes don't break existing functionality.

## Documentation

Update documentation for any new features or changes to existing functionality. This includes:
- Code comments
- Function/method docstrings
- README updates if necessary

## Questions?

If you have any questions about contributing, please reach out to the project maintainers.

Thank you for contributing to AGILE!
