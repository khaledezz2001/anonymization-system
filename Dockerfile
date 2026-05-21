FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ===== CRITICAL: Kill FlashInfer JIT before anything imports vLLM =====
ENV FLASHINFER_DISABLE_JIT=1
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV VLLM_GDN_PREFILL_BACKEND=triton

RUN pip uninstall -y torchvision torchaudio 2>/dev/null || true

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ===== DOWNLOAD ONLY — no GPU needed =====
RUN python3 -u <<'EOF'
from huggingface_hub import snapshot_download
print("Downloading Qwen/Qwen3.6-27B...", flush=True)
snapshot_download(
    repo_id="Qwen/Qwen3.6-27B",
    local_dir="/app/models/Qwen3.6-27B",
    local_dir_use_symlinks=False,
    resume_download=True
)
print("Download complete", flush=True)
EOF

# ===== Clean up disk space =====
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip cache purge

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT ["python3"]
CMD ["-u", "handler.py"]
