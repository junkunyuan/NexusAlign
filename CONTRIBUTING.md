# Contributing to NexusAlign

Thank you ❤️ for your interest in contributing to NexusAlign!

Contributions of all kinds are welcome 👋 — including bug reports, new features, and suggestions.

Please read the following guidelines before contributing.

## 📚 Table of Contents

- [Contributing to NexusAlign](#contributing-to-nexusalign)
  - [📚 Table of Contents](#-table-of-contents)
  - [🐛 Reporting bugs](#-reporting-bugs)
  - [🔀 Submitting Pull Requests](#-submitting-pull-requests)
    - [🛠️ Development Setup](#️-development-setup)
    - [🧹 Code Style](#-code-style)
    - [✅ Testing](#-testing)
    - [📖 Documentation](#-documentation)
    - [🧪 Research Contributions](#-research-contributions)
  - [🤝 Community Standards](#-community-standards)
  - [❓ Questions](#-questions)

## 🐛 Reporting bugs

Before opening an issue:

1. Search existing issues to avoid duplicates.
2. Make sure you are using the latest version.
3. Provide a minimal, reproducible example if reporting a bug.

A good issue should include:

- A clear description of the problem.
- Expected vs. actual behavior.
- Environment details (Python version, torch version, OS, etc.).
- A minimal code snippet or configuration (if applicable).

💡 Well-written issues save time for both maintainers and contributors.

## 🔀 Submitting Pull Requests

To increase the chance of your PR being reviewed and merged quickly:

1. Open an issue first for non-trivial changes.
2. Keep PRs focused — one logical change per PR.
3. Write clear, descriptive commit messages.
4. Ensure all tests pass.
5. Run formatting and linting tools before submitting.

### 🛠️ Development Setup

```bash
pip install -e ".[dev]"
```

### 🧹 Code Style

We use `ruff` for formatting. Before committing, please run:

```bash
make format-check  # get format issue feedback
make format-fixup  # directly fix format issues
```

### ✅ Testing

If your PR introduces new functionality or changes behavior:

1. Add appropriate tests.
2. Ensure existing tests pass.
3. Avoid breaking backward compatibility unless it has been discussed and agreed upon.

### 📖 Documentation

Good documentation is as valuable as good code.

Documentation contributions are highly appreciated, including:

1. Fixing unclear explanations.
2. Improving docstrings.
3. Adding usage examples.
4. Clarifying research assumptions or design choices.

### 🧪 Research Contributions

If you are contributing a new algorithm, model, or training method, please include:

1. A short conceptual explanation.
2. Links to relevant papers (if applicable).
3. Design motivation and key trade-offs.
4. Reproducible configuration (e.g., config files, commands, or scripts).

💡 We prefer implementations that are modular, well-documented, and easy to ablate.

## 🤝 Community Standards

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

🙂 We aim to maintain a respectful, inclusive, and professional environment.


## ❓ Questions

If you're unsure whether a contribution aligns with the project direction, or you need guidance on where to start:

- Feel free to open a discussion.
- You can also start with smaller issues labeled as good first tasks (if available).

🚀 We deeply appreciate your interest in improving NexusAlign and look forward to collaborating with you.
