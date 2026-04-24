# Contributing to ComfyUI HunyuanWorld

Thank you for your interest in contributing to ComfyUI HunyuanWorld!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/comfyui-hunyuan-world.git
   ```
3. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```

2. Install HunyuanWorld 1.0 (for full testing):
   ```bash
   git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
   cd HunyuanWorld-1.0 && pip install -e .
   ```

3. Run tests:
   ```bash
   python -m pytest test/ -v
   ```

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update the version number in `__init__.py` and `pyproject.toml`
5. Create a Pull Request with a clear description

## Reporting Issues

- Use the GitHub issue templates
- Include ComfyUI console output
- Specify your environment (OS, Python version, CUDA version, GPU model)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
