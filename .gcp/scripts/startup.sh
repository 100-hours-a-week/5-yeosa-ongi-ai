#!/bin/bash
set -e

export IMAGE_TAG=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/IMAGE_TAG)
export PROJECT_ID=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/PROJECT_ID)
export APP_ENV=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/APP_ENV)

echo "[INFO] IMAGE_TAG=${IMAGE_TAG}"
echo "[INFO] PROJECT_ID=${PROJECT_ID}"
echo "[INFO] APP_ENV=${APP_ENV}"

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

echo "gcloud CLI 설치"
apt-get install -y apt-transport-https ca-certificates gnupg curl
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
  | tee /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update && apt-get install -y google-cloud-sdk

echo "GCP Artifact Registry 인증"
gcloud auth configure-docker asia-northeast3-docker.pkg.dev --quiet

echo "ai-cpu 이미지 Pull"
docker pull asia-northeast3-docker.pkg.dev/dev-ongi-3-tier/dev-ongi-ai-repo/ai-cpu:${IMAGE_TAG}

echo "FastAPI 앱 컨테이너 실행"
docker run -d \
  --name ai-cpu \
  -e APP_ENV="$APP_ENV" \
  -e PROJECT_ID="$PROJECT_ID" \
  -p 8000:8000 \
  --restart unless-stopped \
  asia-northeast3-docker.pkg.dev/dev-ongi-3-tier/dev-ongi-ai-repo/ai-cpu:${IMAGE_TAG}