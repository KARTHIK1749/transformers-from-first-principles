# Contributing to Transformers from First Principles

Thank you for your interest in contributing!

The goal of this project is not only to implement Transformers using only NumPy but also to make Transformers understandable to everyone through detailed explanations, mathematical derivations, and clean implementations.

---

## Project Philosophy

Every contribution should satisfy three questions:

1. **Why does this component exist?**
2. **How does it work mathematically?**
3. **How is it implemented using NumPy?**

If any of these questions remain unanswered, the contribution is considered incomplete.

---

# Repository Principles

This repository follows a few strict rules.

- Only NumPy is allowed for the Transformer implementation.
- No PyTorch.
- No TensorFlow.
- No JAX.
- No automatic differentiation libraries.
- Every mathematical equation should be explained.
- Every implementation should be beginner-friendly.
- Every Python module should be independently executable whenever possible.

---

# Code Style

Please follow these guidelines:

- Follow PEP 8.
- Use meaningful variable names.
- Add docstrings to all public classes and functions.
- Keep functions focused on a single responsibility.
- Avoid unnecessary optimizations that reduce readability.

---

# Documentation Style

Each documentation chapter should follow this structure:

1. Motivation
2. Intuition
3. Mathematical Derivation
4. Numerical Example
5. NumPy Implementation
6. Complexity Analysis
7. Common Mistakes
8. Summary

Maintaining consistency across chapters is important.

---

# Pull Request Checklist

Before submitting a pull request, ensure that:

- Code runs without errors.
- Unit tests pass.
- Mathematical explanations are correct.
- Documentation has been updated if necessary.
- New modules include example usage when applicable.

---

# Reporting Issues

If you discover a bug, please open an issue including:

- A clear description
- Expected behavior
- Actual behavior
- Steps to reproduce
- Python version
- NumPy version

---

Thank you for helping make this project a valuable educational resource for the community.