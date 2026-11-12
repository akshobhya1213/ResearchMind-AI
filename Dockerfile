# Hugging Face Spaces (Docker SDK) compatible: listens on 7860, runs as UID 1000.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 PRELOAD_MODELS=1 \
    HF_HOME=/home/user/.cache/huggingface DATA_DIR=/home/user/app/data

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY --chown=user . .
RUN python scripts/download_models.py && mkdir -p ${DATA_DIR}

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1
CMD ["python", "app.py"]
