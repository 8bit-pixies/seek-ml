format:
    isort .
    black .

lint:
    ruff check . --fix
    pyright

test:
    pytest --cov seek_ml tests