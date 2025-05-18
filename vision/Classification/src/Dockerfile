# 베이스 이미지로 CUDA 11.1.1 및 cuDNN 8이 포함된 이미지를 사용합니다.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# 필요한 시스템 패키지를 설치합니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 환경 변수 설정
ENV PATH=/opt/conda/bin:$PATH

# conda 업데이트
RUN conda update -n base -c defaults conda

# 환경 파일을 컨테이너로 복사합니다.
COPY environment.yml .

# conda 환경 생성
RUN conda env create -f environment.yml

# 환경 활성화
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# 필요에 따라 추가 파일을 복사합니다.
# COPY . .

# 컨테이너 실행 시 기본 명령을 설정합니다.
CMD ["bash"]
