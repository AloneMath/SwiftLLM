# Contributing

Thanks for contributing to SwiftLLM.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install -e .[dev]
```

## Pull Request Checklist

1. Keep changes focused and small.
2. Add or update tests when behavior changes.
3. Run tests before opening a PR:

```powershell
pytest -q
```

4. Run lint checks:

```powershell
ruff check .
```

5. Update `README.md` or docs if usage changed.

## Language and Style Rules

1. Use English for source code, comments, commit messages, and docs.
2. Do not add Chinese text in code or documentation.
3. Keep naming and CLI help text clear and consistent.

## Data and Artifacts

Training data, checkpoints, and generated artifacts are not tracked by Git.
Do not commit files under ignored directories such as `base_data/`,
`checkpoints/`, `artifacts/`, `data/`, `data_eval/`, and `reports/`.
