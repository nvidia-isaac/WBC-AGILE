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

## Testing

Please ensure that your code changes include appropriate tests. Run existing tests to make sure your changes don't break existing functionality.

## Documentation

Update documentation for any new features or changes to existing functionality. This includes:
- Code comments
- Function/method docstrings
- README updates if necessary

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

## Questions?

If you have any questions about contributing, please reach out to the project maintainers.

Thank you for contributing to AGILE!
