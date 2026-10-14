FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastmcp==4.0.11 hypercorn==0.18.0 httpx==0.28.1

COPY infra/kb-bridge/bridge.py /opt/kb-bridge/bridge.py

CMD ["python", "/opt/kb-bridge/bridge.py"]
