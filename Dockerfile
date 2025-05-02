FROM python:3.10.17-slim

WORKDIR /app

# 필요한 패키지 설치 (wget 포함)
RUN apt-get update && apt-get install -y \
    git wget \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1 \ 
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# 모델 다운로드
RUN mkdir -p app/model && \
    wget -O app/model/insight-face-v3.pt \
    https://github.com/foamliu/InsightFace-v3/releases/download/v1.0/insight-face-v3.pt

# COPY clip_model/ViT-B-32.pt /root/.cache/clip/ViT-B-32.pt

# PYTHONPATH 설정
ENV PYTHONPATH=/app

# 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
