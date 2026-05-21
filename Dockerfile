FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ===== CRITICAL: Prevent FlashInfer JIT compilation =====
ENV FLASHINFER_DISABLE_JIT=1
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV VLLM_GDN_PREFILL_BACKEND=triton

RUN pip uninstall -y torchvision torchaudio 2>/dev/null || true

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ===== DOWNLOAD MODEL =====
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

# ===== PRE-WARM TORCH.COMPILE & CUDA GRAPHS =====
# This bakes compilation artifacts into the image so they don't happen at request time
ENV VLLM_TORCH_COMPILE_CACHE_DIR=/root/.cache/vllm/torch_compile_cache
RUN python3 -u <<'EOF'
import torch
from vllm import LLM, SamplingParams

print("Pre-warming torch.compile cache...", flush=True)
llm = LLM(
    model="/app/models/Qwen3.6-27B",
    dtype="float16",
    max_model_len=16384,        # MUST match runtime max_model_len
    gpu_memory_utilization=0.90,
    enforce_eager=False,        # Enable CUDA graphs in warmup
)
_ = llm.generate("Warmup prompt for NER", SamplingParams(max_tokens=5))
print("Warmup complete")
del llm
torch.cuda.empty_cache()
EOF

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT ["python3"]
CMD ["-u", "handler.py"]
