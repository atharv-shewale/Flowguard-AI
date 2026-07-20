#!/usr/bin/env bash
set -euo pipefail

FASTAPI_URL="${FASTAPI_URL:-http://127.0.0.1:8000/predict}"
FLASK_URL="${FLASK_URL:-http://127.0.0.1:5000/predict}"

FASTAPI_PAYLOAD='{"age":0.038,"sex":0.05,"bmi":0.061,"bp":0.021,"s1":-0.044,"s2":-0.034,"s3":-0.043,"s4":-0.002,"s5":0.019,"s6":-0.017}'

echo "Testing FastAPI endpoint: $FASTAPI_URL"
curl -sS -X POST "$FASTAPI_URL" \
  -H "Content-Type: application/json" \
  -d "$FASTAPI_PAYLOAD" | tee /tmp/fastapi_smoke_result.json

echo
echo "Testing Flask endpoint: $FLASK_URL"
curl -sS -X POST "$FLASK_URL" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data "age=0.038&sex=0.05&bmi=0.061&bp=0.021&s1=-0.044&s2=-0.034&s3=-0.043&s4=-0.002&s5=0.019&s6=-0.017" \
  -o /tmp/flask_smoke_result.html

echo "Smoke tests completed."
echo "FastAPI response saved to /tmp/fastapi_smoke_result.json"
echo "Flask response saved to /tmp/flask_smoke_result.html"
