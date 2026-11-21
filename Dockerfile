FROM python:3.8-slim

WORKDIR /app

# libs systeme dont mxnet a besoin
RUN apt-get update && apt-get install -y libgomp1 libquadmath0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# le code est monte en volume pendant le dev (voir README)
ENV PYTHONPATH=/app/src
