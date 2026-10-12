FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY app ./app
RUN mkdir -p data/raw data/processed models
RUN uv run creditrisk-pipeline
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
