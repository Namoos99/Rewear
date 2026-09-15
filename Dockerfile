FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY scripts ./scripts
COPY tests ./tests
CMD ["python", "scripts/evaluate.py", "--data", "data/sample"]
