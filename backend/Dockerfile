FROM python:3.12-slim AS runtime

# System deps required by TensorFlow on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY plant_disease ./plant_disease
COPY wsgi.py .

COPY artifacts ./artifacts

# Environment defaults
ENV BACKBONE=efficientnetv2b0
ENV IMG_SIZE=256
ENV USE_TTA=true
ENV CONFIDENCE_THRESHOLD=0.6
ENV TOP_K=3
ENV MAX_UPLOAD_MB=10

ENV HOST=0.0.0.0
ENV PORT=5001
ENV WORKERS=2

# Silence TensorFlow startup noise
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0

# Non-root user
RUN addgroup --system plantai && adduser --system --ingroup plantai plantai && chown -R plantai:plantai /app
USER plantai

EXPOSE ${PORT}

# Entrypoint
CMD gunicorn --bind 0.0.0.0:${PORT} --timeout 180 --workers ${WORKERS} wsgi:app
