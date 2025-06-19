#!/bin/bash
set -e

echo "Docker 설치"
apt-get update
apt-get install -y docker.io

echo "Docker 데몬 시작 중"
systemctl start docker
systemctl enable docker

echo "node_exporter 컨테이너 실행"
docker run -d \
  --name node_exporter \
  --net=host \
  --pid=host \
  --restart unless-stopped \
  prom/node-exporter

echo "GCP Artifact Registry 인증"
gcloud auth configure-docker asia-northeast3-docker.pkg.dev --quiet

echo "ai-cpu 이미지 Pull"
docker pull asia-northeast3-docker.pkg.dev/dev-ongi-3-tier/dev-ongi-ai-repo/ai-cpu:${IMAGE_TAG}

echo "FastAPI 앱 컨테이너 실행"
docker run -d \
  --name ai-cpu \
  -e PROJECT_ID="$PROJECT_ID" \
  -p 8000:8000 \
  --restart unless-stopped \
  asia-northeast3-docker.pkg.dev/dev-ongi-3-tier/dev-ongi-ai-repo/ai-cpu:${IMAGE_TAG}