PY ?= .venv/bin/python
CONFIG ?= automation/config.yaml

.PHONY: setup models run forecast fetch serve test docker-build docker-up docker-run clean

setup:            ## create venv, install deps, download models, smoke test
	./scripts/setup.sh

models:           ## download all model weights
	$(PY) -m automation.kronos_auto setup-models --all

run:              ## forecast every enabled symbol in $(CONFIG)
	$(PY) -m automation.kronos_auto run --config $(CONFIG)

forecast:         ## make forecast CSV=data/foo.csv [PRED_LEN=24]
	$(PY) -m automation.kronos_auto forecast --csv $(CSV) $(if $(PRED_LEN),--pred-len $(PRED_LEN))

fetch:            ## make fetch SOURCE=yfinance SYMBOL=AAPL INTERVAL=1d
	$(PY) -m automation.kronos_auto fetch --source $(or $(SOURCE),yfinance) --symbol $(SYMBOL) --interval $(or $(INTERVAL),1d)

serve:            ## start the web UI on :7070
	$(PY) -m automation.kronos_auto serve

test:             ## run regression tests
	$(PY) -m pytest tests -q

docker-build:
	docker compose build

docker-up:        ## web UI in docker on :7070
	docker compose up -d web

docker-run:       ## one forecast run in docker
	docker compose run --rm forecaster

clean:
	rm -rf outputs __pycache__ */__pycache__
