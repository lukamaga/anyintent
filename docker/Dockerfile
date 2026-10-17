# AnyIntent HTTP server, CPU image (Python slim, non-root user).
# Build from the library root, the folder with pyproject.toml:
#   docker build -f docker/Dockerfile -t anyintent-server:cpu .
# Run (the Hugging Face token is given at run time and never stored in the image):
#   docker run --rm -p 127.0.0.1:8000:8000 -e HF_TOKEN -v anyintent-data:/data anyintent-server:cpu
# The model (ANYINTENT_MODEL, default lukashm/r2intent-v1, a gated repository) is downloaded at the first start into
# the volume /data (Hugging Face cache in /data/hf, saved intent lists in /data/lists).
# For a reproducible build pin the base image by its digest (docker buildx imagetools inspect python:3.12-slim-bookworm
# prints it): --build-arg PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:<digest>
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

# CPU-only PyTorch wheels (much smaller than the default CUDA wheels)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# the model revision the server loads (a tag or a commit hash; empty: the repository's default branch)
ARG ANYINTENT_REVISION=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    ANYINTENT_MODEL=lukashm/r2intent-v1 \
    ANYINTENT_REVISION=${ANYINTENT_REVISION} \
    ANYINTENT_LISTS_DIR=/data/lists

RUN groupadd --system --gid 10001 anyintent \
 && useradd --system --uid 10001 --gid anyintent --create-home --home-dir /home/anyintent anyintent \
 && mkdir -p /data/hf /data/lists \
 && chown -R anyintent:anyintent /data

WORKDIR /opt/anyintent
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY docker/constraints.txt ./constraints.txt
COPY src ./src
RUN pip install --index-url "${TORCH_INDEX_URL}" "$(grep '^torch==' constraints.txt)" \
 && pip install -c constraints.txt "./[server]" \
 && rm -rf build src/*.egg-info

USER anyintent
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Options after the image name are added to these (for example --revision TAG or --max-batch 32); the API key comes
# from the environment variable ANYINTENT_API_KEY (docker run -e ANYINTENT_API_KEY ...). Without a key the server
# accepts only requests addressed to 127.0.0.1 or localhost (a guard against DNS rebinding); the health check uses
# 127.0.0.1.
ENTRYPOINT ["anyintent-serve", "--host", "0.0.0.0", "--port", "8000"]
CMD []
