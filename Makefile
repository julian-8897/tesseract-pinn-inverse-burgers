.PHONY: compile lint format test build-tesseracts train-posterior smoke benchmark run-cli run-app

compile:
	uv run python -m py_compile app.py inverse_problem.py configs.py fmpe_posterior.py

lint:
	uv run --with ruff ruff check .
	uv run --with ruff ruff format --check .

format:
	uv run --with ruff ruff check --fix .
	uv run --with ruff ruff format .

test:
	uv run --with pytest python -m pytest tests -q

build-tesseracts:
	./buildall.sh

train-posterior:
	uv run python fmpe_posterior.py train --n-sims 10000

smoke:
	uv run --with pytest python -m pytest tests/test_container_smoke.py -q

benchmark:
	uv run python inverse_problem.py --backend both --epochs 100 --seed 123 --out runs

run-cli:
	uv run python inverse_problem.py --backend both --epochs 50 --seed 123

run-app:
	uv run streamlit run app.py
