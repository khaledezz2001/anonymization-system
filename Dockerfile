FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV TRANSFORMERS_CACHE=/runpod-volume/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Python deps + fast download tool
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt hf_transfer

# No model download during build — model lives on Network Volume
# It will auto-download on first start

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
