FROM python:3.12-slim
RUN python -m venv /opt/openhands-sdk-venv && /opt/openhands-sdk-venv/bin/pip install --no-cache-dir openhands-sdk openhands-tools fastapi
WORKDIR /app
COPY data /app/data
