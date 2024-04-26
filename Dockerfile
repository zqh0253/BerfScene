# FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/home/user/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

# apt install by root user
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

WORKDIR /home/user

# RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

COPY --chown=user . /home/user
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN wget https://www.dropbox.com/scl/fi/105qy7mkqfjcmnfd3tmv0/edit.pth?rlkey=qcd67cdrqz4jra0p3er966iuk -O clevr.pth

RUN wget https://www.dropbox.com/scl/fi/k5qc5y5rmhuru5eztegbn/gradio_draggable-0.0.1-py3-none-any.whl?rlkey=fr36c5gfht4d8wwjr0bb9qu9w -O gradio_draggable-0.0.1-py3-none-any.whl
RUN pip install gradio_draggable-0.0.1-py3-none-any.whl

ENV TORCH_EXTENSIONS_DIR=/home/user/.cache


# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["python", "app.py"]
