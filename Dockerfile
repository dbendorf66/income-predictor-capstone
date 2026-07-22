# Runs the Streamlit LLM interface against an already-trained model.
#
#   docker build -t income-predictor .
#   docker run -p 8501:8501 --env-file .env income-predictor
#
# The image bundles models/best_model.joblib, so it serves predictions without
# needing the raw dataset. To retrain inside the container instead:
#   docker run --env-file .env -v "$(pwd)/data:/app/data" income-predictor \
#     python -m src.train
FROM python:3.13-slim

WORKDIR /app

# Install dependencies first so the layer caches across source edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
