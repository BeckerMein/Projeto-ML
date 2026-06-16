FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY app ./app
COPY src ./src
COPY artifacts/modeling/20260613-134558 ./artifacts/modeling/20260613-134558
COPY README.md ./

EXPOSE 5000 8501

CMD ["streamlit", "run", "app/app_dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
