FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Python deps + fast download
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt hf_transfer

# ===============================
# DOWNLOAD Qwen/Qwen3-30B-A3B-Instruct-2507
# ===============================
RUN python3 -u <<'EOF'
from huggingface_hub import snapshot_download

print("Downloading Qwen/Qwen3-30B-A3B-Instruct-2507...", flush=True)

snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    local_dir="/models/qwen",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Qwen/Qwen3-30B-A3B-Instruct-2507 download complete", flush=True)
EOF

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
