# torch_ga Documentation

This directory contains the Sphinx documentation for torch_ga.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
uv sync --extra docs
```

Or with pip:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

### Build HTML

```bash
cd docs
make html
```

The generated HTML documentation will be in `build/html/`.

### View Locally

Open `build/html/index.html` in your browser:

```bash
# Linux/Mac
open build/html/index.html

# Or with Python
python -m http.server -d build/html 8000
```

Then navigate to `http://localhost:8000`

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── api/                 # API reference
│   │   ├── index.rst
│   │   ├── geometric_algebra.rst
│   │   ├── multivector.rst
│   │   └── layers.rst
│   └── tutorials/           # User guides
│       ├── quickstart.rst
│       └── cuda.rst
├── Makefile                 # Build automation
└── build/                   # Generated output (git-ignored)
```

## Other Build Targets

```bash
# Build PDF (requires LaTeX)
make latexpdf

# Clean build artifacts  
make clean

# Check links
make linkcheck

# Show all available targets
make help
```

## Contributing

When adding new documentation:

1. Add new `.rst` files to `source/` or subdirectories
2. Update `index.rst` or relevant toctree to include the new file
3. Run `make html` to verify it builds correctly
4. Check the output in `build/html/`

## Auto-documentation

API documentation is automatically generated from docstrings using Sphinx's autodoc extension. Ensure your code has proper docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """Short description.
    
    Longer description with more details.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
    """
    pass
```

## Hosting

Documentation can be hosted on:

- **Read the Docs**: Connect your GitHub repo to automatically build and host docs
- **GitHub Pages**: Use GitHub Actions to build and deploy to gh-pages branch
- **Custom hosting**: Deploy the `build/html/` directory to any web server
