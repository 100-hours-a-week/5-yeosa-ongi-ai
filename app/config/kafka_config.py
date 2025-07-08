import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL")

KAFKA_GROUP_ID_MAP = {
    "album.ai.category.request": os.getenv("KAFKA_GROUP_CATEGORY"),
    "album.ai.duplicate.request": os.getenv("KAFKA_GROUP_DUPLICATE"),
    "album.ai.quality.request": os.getenv("KAFKA_GROUP_QUALITY"),
    "album.ai.score.request": os.getenv("KAFKA_GROUP_SCORE"),
    "album.ai.embedding.request": os.getenv("KAFKA_GROUP_EMBEDDING"),
    "album.ai.people.request": os.getenv("KAFKA_GROUP_PEOPLE"),
}

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
