# tswheel

## Project Structure
tswheel/
├── README.md
├── LICENSE            # Open-source license
├── CHANGELOG.md       # Changelog with format based on https://keepachangelog.com/en/1.1.0/
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── .env               # Python-dotenv reads key-value pairs from .env and sets them as environment variables
├── docs/              # User-written project functionality and usage docs
├── output_data/       # Data generated from scripts
├── input_data/        # Static, externally obtained input data files
├── models/            # Trained and serialized models and model details
├── references/        # Static, external reference documents 
├── reports/           # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/       # Generated graphics and figures to be used in reporting
├── scripts/
├── notebooks/         # Jupyter notebooks
├── src/
│   └── tswheel/
│       └── __init__.py
└── tests/
