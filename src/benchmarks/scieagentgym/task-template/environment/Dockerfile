FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && \
    mkdir -p /opt/openhands-sdk-venv && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    . "$HOME/.local/bin/env" && \
    uv python install 3.12 && \
    uv venv /opt/openhands-sdk-venv --python 3.12 --clear && \
    . /opt/openhands-sdk-venv/bin/activate && \
    uv pip install openhands-sdk openhands-tools fastapi

RUN pip install --no-cache-dir numpy scipy sympy pandas scikit-learn matplotlib
WORKDIR /app
COPY data /app/data
