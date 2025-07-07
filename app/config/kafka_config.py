import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID")
KAFKA_GPU_GROUP_ID = os.getenv("KAFKA_GPU_GROUP_ID")

if not KAFKA_BROKER_URL:
    raise EnvironmentError("KAFKA_BROKER_URL를 설정해야 합니다.")
if not KAFKA_GROUP_ID or not KAFKA_GPU_GROUP_ID:
    raise EnvironmentError("KAFKA 그룹 ID를 모두 설정해야 합니다.")

KAFKA_REQUEST_TOPICS = [
    "album.ai.embedding.request",
    "album.ai.duplicate.request",
    "album.ai.quality.request",
    "album.ai.category.request",
    "album.ai.score.request",
    "album.ai.people.request",
]

KAFKA_RESPONSE_TOPIC_MAP = {
    req: req.replace("request", "response")
    for req in KAFKA_REQUEST_TOPICS
}
