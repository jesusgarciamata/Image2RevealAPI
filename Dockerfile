FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    SAM2_CHECKPOINT=/app/models/sam2.1_hiera_tiny.pt \
    SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}+cpu" "torchvision==${TORCHVISION_VERSION}+cpu"

ARG SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
RUN SAM2_BUILD_CUDA=0 pip install --no-cache-dir --no-deps \
    "https://github.com/facebookresearch/sam2/archive/${SAM2_COMMIT}.zip"

ARG SAM2_CHECKPOINT_SHA256=7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
RUN mkdir -p /app/models \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt', '/app/models/sam2.1_hiera_tiny.pt')" \
    && echo "${SAM2_CHECKPOINT_SHA256}  /app/models/sam2.1_hiera_tiny.pt" | sha256sum --check

COPY app ./app
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
