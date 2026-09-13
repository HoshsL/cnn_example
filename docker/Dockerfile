ARG PYTORCH="2.9.1"
ARG CUDA="13.0"
ARG CUDNN="9"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        git \
        git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD [ "/bin/bash" ]
