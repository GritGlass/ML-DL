FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    git \
    bash \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 환경 변수 설정
ENV PATH=/opt/conda/bin:$PATH

# conda 업데이트
RUN conda update -n base -c defaults conda

# environment.yml 파일 복사
COPY environment.yml .

# conda 환경 생성
RUN conda env create -f environment.yml

SHELL ["/bin/bash", "-c"]

# Conda 초기화 및 환경 생성
RUN conda init bash && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate draem_test" >> ~/.bashrc && \
    bash -c "source /opt/conda/etc/profile.d/conda.sh"

# Git 리포지토리 클론
RUN git clone https://github.com/VitjanZ/DRAEM.git

# 리포지토리로 이동
WORKDIR /app/DRAEM

# 스크립트 실행
RUN chmod +x scripts/download_dataset.sh
RUN bash scripts/download_dataset.sh

RUN chmod +x scripts/download_pretrained.sh
RUN bash scripts/download_pretrained.sh

# 기본 실행 명령
CMD ["bash", "-l"]