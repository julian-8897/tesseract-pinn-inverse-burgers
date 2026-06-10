.PHONY: compile lint format test test-slow build-tesseracts train-posterior posterior-calibration plot-sbi plot-pinn smoke benchmark run-cli run-app

compile:
	uv run python -m compileall -q app.py src/burgers_inverse scripts tests tesseracts

lint:
	uv run --with ruff ruff check .
	uv run --with ruff ruff format --check .

format:
	uv run --with ruff ruff check --fix .
	uv run --with ruff ruff format .

test:
	uv run --with pytest python -m pytest tests -q -m "not slow"

test-slow:
	uv run --with pytest python -m pytest tests -q -m slow

build-tesseracts:
	./buildall.sh

train-posterior:
	uv run burgers-fmpe train --n-sims 10000

posterior-calibration:
	uv run python -m scripts.fmpe_diagnostics calibrate

plot-sbi: posterior-calibration
	uv run python -m scripts.plot_sbi_results \
		--calibration-report artifacts/fmpe_calibration.json

plot-pinn:
	uv run python -m scripts.regenerate_figures \
		--epochs 100 --seed 123 --nx 160 --nt 90

smoke:
	uv run --with pytest python -m pytest tests/test_container_smoke.py -q

benchmark:
	uv run burgers-inverse --backend both --epochs 100 --seed 123 --out runs

run-cli:
	uv run burgers-inverse --backend both --epochs 50 --seed 123

run-app:
	uv run streamlit run app.py
