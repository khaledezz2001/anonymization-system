FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Qwen3.6-27B — dense 27B model, superior multilingual NER for Russian, Greek, etc.

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Remove pre-installed torchvision/torchaudio — not needed for text-only LLM inference
RUN pip uninstall -y torchvision torchaudio 2>/dev/null || true

# Install dependencies (includes vLLM for inference)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt hf_transfer

# Download model weights into the image (no network volume needed)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3.6-27B', local_dir='/app/models/Qwen3.6-27B', local_dir_use_symlinks=False)"

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
