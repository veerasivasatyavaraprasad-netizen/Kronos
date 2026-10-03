FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    KRONOS_NO_BROWSER=1 \
    MPLBACKEND=Agg

WORKDIR /app

# CPU-only PyTorch keeps the image small (falls back to the default PyPI wheel if that index is unreachable).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url ${TORCH_INDEX_URL} || pip install torch
COPY requirements.txt requirements-app.txt ./
RUN pip install -r requirements-app.txt

COPY model ./model
COPY automation ./automation
COPY webui ./webui
COPY data ./data
COPY tests ./tests
COPY finetune_csv ./finetune_csv
COPY conftest.py ./

# Bake the default models into the image so containers start offline.
RUN python -m automation.kronos_auto setup-models --model kronos-small \
 && python -m automation.kronos_auto setup-models --model kronos-mini

# Writable for hosts that run the container as a non-root user (e.g. Hugging Face Spaces).
RUN mkdir -p /app/outputs /app/webui/prediction_results && chmod -R a+rwX /app /models

ENV PORT=7070 KRONOS_AUTOLOAD_MODEL=kronos-small
EXPOSE 7070
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
# One worker keeps a single model copy in memory; threads serve concurrent requests.
CMD gunicorn --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:${PORT} webui.app:app
