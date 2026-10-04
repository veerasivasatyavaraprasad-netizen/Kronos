PY ?= .venv/bin/python
CONFIG ?= automation/config.yaml

.PHONY: luna luna-cycle luna-backtest luna-test luna-check luna-docker setup models run forecast fetch serve test examples finetune finetune-full qlib-pipeline docker-build docker-up docker-run clean

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

examples:         ## run the example scripts that work offline (charts -> outputs/)
	cd /tmp && MPLBACKEND=Agg $(abspath $(PY)) $(CURDIR)/examples/prediction_example.py
	cd /tmp && MPLBACKEND=Agg $(abspath $(PY)) $(CURDIR)/examples/prediction_wo_vol_example.py
	cd /tmp && MPLBACKEND=Agg $(abspath $(PY)) $(CURDIR)/examples/prediction_batch_example.py

finetune:         ## quick CPU fine-tune on the bundled CSV (minutes)
	$(PY) -m automation.kronos_auto finetune --config finetune_csv/configs/config_quick_cpu.yaml

finetune-full:    ## full CSV fine-tune (GPU recommended); GPUS=2 for multi-GPU
	$(PY) -m automation.kronos_auto finetune --config finetune_csv/configs/config_ali09988_candle-5min.yaml --gpus $(or $(GPUS),1)

qlib-pipeline:    ## Qlib A-share pipeline: download, preprocess, fine-tune, backtest
	PYTHON=$(abspath $(PY)) ./finetune/run_pipeline.sh

docker-build:
	docker compose build

docker-up:        ## web UI in docker on :7070
	docker compose up -d web

docker-run:       ## one forecast run in docker
	docker compose run --rm forecaster

clean:
	rm -rf outputs __pycache__ */__pycache__

# ---------------------------------------------------------------- LunaTrade
luna:             ## LunaTrade engine + dashboard on :8800 (mode from lunatrade.yaml)
	$(PY) -m lunatrade run

luna-cycle:       ## one decision cycle on live market data
	$(PY) -m lunatrade cycle

luna-backtest:    ## backtest + walk-forward + Monte Carlo + stress on Binance data
	$(PY) -m lunatrade backtest --symbols BTCUSDT,ETHUSDT,SOLUSDT --bars 2000 --walk-forward --monte-carlo --stress

luna-test:        ## LunaTrade test suite
	$(PY) -m pytest tests/lt -q

luna-check:       ## which keys/services are configured (never prints secrets)
	$(PY) -m lunatrade check

luna-docker:      ## postgres + redis + lunatrade in docker
	docker compose -f docker-compose.lunatrade.yml up -d --build
