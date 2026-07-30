FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY plant_disease ./plant_disease
COPY artifacts ./artifacts
COPY wsgi.py .

ENV PORT=7860
ENV BACKBONE=efficientnetb0
ENV USE_TTA=true

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "180", "--workers", "1", "wsgi:app"]
