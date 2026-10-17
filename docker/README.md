# AnyIntent server in Docker

Two images of the HTTP server (`anyintent-serve`, see [docs/server.md](../docs/server.md) of the library):

- `Dockerfile`: CPU, on `python:3.12-slim-bookworm`, with the CPU build of PyTorch.
- `Dockerfile.cuda`: NVIDIA GPU, on the official `nvidia/cuda` runtime image, with a CUDA build of PyTorch.

Both run as a non-root user (uid 10001), listen on port 8000 inside the container and keep their data in the volume
`/data`: the Hugging Face cache (`/data/hf`, the model is downloaded there at the first start) and the saved intent
lists (`/data/lists`). No model and no secret is in an image.

These Docker files are provided as they are: they were not built or tested for the release v0.1.0.

## Build

From the library root (the folder with `pyproject.toml`):

```bash
docker build -f docker/Dockerfile -t anyintent-server:cpu .
docker build -f docker/Dockerfile.cuda -t anyintent-server:cuda .
```

Build arguments: `PYTHON_IMAGE` and `TORCH_INDEX_URL` (CPU image); `CUDA_IMAGE` and `TORCH_INDEX_URL` (CUDA image,
default CUDA 12.6 for both; change them together); `ANYINTENT_REVISION` in both (the model revision the server loads,
a tag or a commit hash; empty by default, which loads the repository's default branch).

Both images install the Python packages at the versions in `docker/constraints.txt`. For a build that does not change
over time, also pin the base image by its digest; `docker buildx imagetools inspect python:3.12-slim-bookworm` prints
it:

```bash
docker build -f docker/Dockerfile -t anyintent-server:cpu \
    --build-arg PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:DIGEST \
    --build-arg ANYINTENT_REVISION=REVISION .
```

## Run

The model `lukashm/r2intent-v1` is gated on Hugging Face: accept its terms on its model page with your account, then
pass a read token at run time. `-e HF_TOKEN` copies the variable from your shell, so the token is not written in the
command:

```bash
export HF_TOKEN=hf_...            # your Hugging Face read token
docker run --rm -p 127.0.0.1:8000:8000 -e HF_TOKEN -v anyintent-data:/data anyintent-server:cpu
docker run --rm --gpus all -p 127.0.0.1:8000:8000 -e HF_TOKEN -v anyintent-data:/data anyintent-server:cuda
```

`-p 127.0.0.1:8000:8000` makes the server reachable from this machine only. Without an API key the server also
accepts only requests addressed to `127.0.0.1` or `localhost` (a guard against DNS rebinding from web pages; a request
for another name gets a 400 that names `--allowed-host`). To serve other machines, set an API key, which accepts any
host name, and publish the port as you need it:

```bash
export ANYINTENT_API_KEY="$(openssl rand -hex 24)"
docker run --rm -p 8000:8000 -e HF_TOKEN -e ANYINTENT_API_KEY -v anyintent-data:/data anyintent-server:cpu
curl -H "Authorization: Bearer $ANYINTENT_API_KEY" http://localhost:8000/health
```

Options after the image name are passed to `anyintent-serve`, for example a pinned model revision (replace REVISION
with a tag or a commit hash) and smaller limits:

```bash
docker run --rm -p 127.0.0.1:8000:8000 -e HF_TOKEN -v anyintent-data:/data anyintent-server:cpu \
    --revision REVISION --max-batch 32 --max-text-chars 1000
```

`--max-concurrency` (default 64) caps the requests in progress; more get a 503 `busy`, and `/health` is not counted.

Another model: `-e ANYINTENT_MODEL=USER/REPOSITORY`. After the first start the model is in the volume; with
`-e HF_HUB_OFFLINE=1` the server then starts without contacting Hugging Face.

The health check of the image calls `GET /health`, which answers without the API key (with only `status` and
`ready`). The first start can take several minutes while the model downloads (the check allows 10 minutes).

A host folder instead of the named volume must be writable by uid 10001, for example
`mkdir -p data && sudo chown 10001:10001 data` and then `-v "$PWD/data:/data"`.

The CUDA image starts with `--device cuda` and stops with an error when the container sees no GPU (it never falls back
to the CPU unnoticed). The host needs an NVIDIA driver that supports the CUDA version of the PyTorch wheels and the
NVIDIA Container Toolkit.
