FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND="noninteractive" \
    MAX_JOBS=8

# install apt dependencies
RUN apt-get update \
    && apt-get install -y openssh-server iputils-ping net-tools iproute2 traceroute netcat \
    build-essential libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev tzdata \
    && apt clean

# install pytorch_memlab
RUN pip install py-spy pytorch_memlab

# reinstall flash-attn
RUN pip install flash-attn --no-build-isolation -U

# install python dependencies
RUN pip install loguru tqdm ninja tensorboard \
        sentencepiece fire tabulate easydict \
        transformers==4.48.1 awscli rpyc pythonping \
        torchvision==0.18.0 hydra-core accelerate \
        redis opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp prometheus-client \
        omegaconf black==22.8.0 mypy-extensions pathspec tensorboardX nvitop antlr4-python3-runtime==4.11.0 \
        ray==2.40.0 deepspeed==0.16.0 vllm==0.6.3 peft

RUN apt update && apt install --fix-broken && apt install -y default-jre-headless openjdk-8-jdk
RUN pip install git+https://github.com/facebookresearch/hydra.git

# set nccl.conf and nccl_env
COPY nccl.conf /etc/nccl.conf

# set time zone to CST
RUN apt update && apt install -y tzdata  \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt clean

# fix deepspeed bug
COPY parameter_offload.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/parameter_offload.py
COPY partitioned_param_coordinator.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py

# launch script
WORKDIR /workspace/