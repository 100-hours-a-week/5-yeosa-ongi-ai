import os
from dotenv import load_dotenv

# 로컬 개발 시 .env 로드
load_dotenv()


KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID")

if KAFKA_BROKER_URL is None:
    raise EnvironmentError("KAFKA_BROKER_URL를 설정해야 합니다.")
if KAFKA_GROUP_ID is None:
    raise EnvironmentError("KAFKA_GROUP_ID를 설정해야 합니다.")


KAFKA_REQUEST_TOPICS = [
    "album.ai.embedding.request",
    "album.ai.duplicate.request",
    "album.ai.quality.request",
    "album.ai.category.request",
    "album.ai.score.request",
    "album.ai.people.request"
]

KAFKA_RESPONSE_TOPICS = [
    "album.ai.embedding.response",
    "album.ai.duplicate.response",
    "album.ai.quality.response",
    "album.ai.category.response",
    "album.ai.score.response",
    "album.ai.people.response"
]

KAFKA_RESPONSE_TOPIC_MAP = {
    req: req.replace("request", "response")
    for req in KAFKA_REQUEST_TOPICS
}
