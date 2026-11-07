# Makefile for C-MAPSS Predictive Maintenance Pipeline

.PHONY: all setup check-data interim features train evaluate threshold stream-demo test lint clean

all: setup check-data interim features train evaluate threshold

setup:
	@echo "Setting up environment..."
	@python -m venv .venv || echo "venv may already exist"
	@.venv\Scripts\activate && pip install -r requirements.txt
	@echo "Setup complete."

check-data:
	@echo "Checking data files..."
	@python scripts\check_data.py
	@echo "Data check complete."

interim:
	@echo "Creating interim data..."
	@python -m src.data.make_interim
	@echo "Interim data created."

features:
	@echo "Engineering features..."
	@python -m src.features.make_features
	@echo "Features created."

train:
	@echo "Training model..."
	@python -m src.modeling.train_tree
	@echo "Training complete."

evaluate:
	@echo "Evaluating model..."
	@python -m src.modeling.evaluate
	@echo "Evaluation complete."

threshold:
	@echo "Calibrating threshold..."
	@python -m src.modeling.threshold
	@echo "Threshold calibrated."

stream-demo:
	@echo "Running streaming prediction demo..."
	@python -m src.modeling.predict --all
	@echo "Streaming demo complete."

test:
	@echo "Running tests..."
	@python -m pytest tests/ -v
	@echo "Tests complete."

lint:
	@echo "Running lint checks..."
	@python -m flake8 src/ --max-line-length=88
	@echo "Lint complete."

clean:
	@echo "Cleaning up..."
	@rm -rf mlruns/ reports/figures/*.png models/*.pkl
	@echo "Clean complete."