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

EXPOSE 7070
CMD ["python", "-m", "automation.kronos_auto", "serve", "--port", "7070"]
