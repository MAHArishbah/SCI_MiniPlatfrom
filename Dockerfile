FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCI_DATA_DIR=/app/data/raw \
    SCI_DB_PATH=/app/warehouse/sci.duckdb \
    SCI_DQ_REPORT_PATH=/app/artifacts/dq_report.json

WORKDIR /app
RUN useradd --create-home --uid 1000 app

# Dependencies first so code changes don't invalidate the (slow) install layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY sci ./sci
COPY dq_check.py ./
COPY data ./data
COPY artifacts ./artifacts
COPY docker/entrypoint.sh /entrypoint.sh
RUN mkdir -p warehouse logs && chown -R app:app /app && chmod +x /entrypoint.sh

ARG GIT_SHA=local
ENV GIT_SHA=${GIT_SHA}

USER app
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=12 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
