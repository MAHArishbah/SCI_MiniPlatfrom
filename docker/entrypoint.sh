#!/bin/sh
# serve: run the DQ check, (re)build the warehouse, then start the API.
# Anything else is executed as-is, e.g. `docker compose run --rm api python -m sci.ml.train`.
set -eu

if [ "${1:-serve}" = "serve" ]; then
  python dq_check.py --input "$SCI_DATA_DIR" --output "$SCI_DQ_REPORT_PATH"
  python -m sci.pipeline.build --input "$SCI_DATA_DIR" --db "$SCI_DB_PATH"
  exec uvicorn sci.api.main:app --host 0.0.0.0 --port 8000 --workers "${WEB_CONCURRENCY:-2}"
fi
exec "$@"
