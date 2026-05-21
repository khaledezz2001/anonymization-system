FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ===== NEW: Prevent JIT compilation issues =====
ENV VLLM_GDN_PREFILL_BACKEND=triton
ENV FLASHINFER_DISABLE_JIT=1

RUN pip uninstall -y torchvision torchaudio 2>/dev/null || true

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN python3 -u <<'EOF'
from huggingface_hub import snapshot_download
print("Downloading openai/gpt-oss-20b...", flush=True)
snapshot_download(
    repo_id="openai/gpt-oss-20b",
    local_dir="/app/models/gpt-oss-20b",
    local_dir_use_symlinks=False,
    resume_download=True
)
print("Download complete", flush=True)
EOF

# ===== NEW: Pre-warm torch.compile cache =====
# This bakes the compilation artifacts into the image
ENV VLLM_TORCH_COMPILE_CACHE_DIR=/root/.cache/vllm/torch_compile_cache
RUN python3 -u <<'EOF'
import torch
from vllm import LLM, SamplingParams

print("Pre-warming torch.compile cache...")
llm = LLM(
    model="/app/models/gpt-oss-20b",
    dtype="auto",
    max_model_len=512,  # Shorter is fine for cache warmup
    gpu_memory_utilization=0.90,
)
_ = llm.generate("Warmup", SamplingParams(max_tokens=5))
print("Warmup complete")
del llm
torch.cuda.empty_cache()
EOF

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT ["python3"]
CMD ["-u", "handler.py"]
