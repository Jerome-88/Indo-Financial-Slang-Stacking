FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Light build: traditional ML only (no torch/transformers = ~300 MB image vs ~3 GB)
COPY requirements_light.txt .
RUN pip install --no-cache-dir -r requirements_light.txt

COPY . .

# HF Spaces default port
EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
