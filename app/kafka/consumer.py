import asyncio
from app.kafka.producer import create_kafka_producer
from app.utils.kafka_utils import create_kafka_consumer, process_partition_batch
from app.kafka.handler import (
    embedding as embedding_handler,
    people as people_handler,
    category as category_handler,
    duplicate as duplicate_handler,
    quality as quality_handler,
    score as score_handler,
)

from app.schemas.kafka import (
    embedding as embedding_schema,
    people as people_schema,
    categories as categories_schema,
    duplicate as duplicate_schema,
    quality as quality_schema,
    score as score_schema,
)
from app.config.kafka_config import KAFKA_BROKER_URL

ALL_TOPICS = [
    "album.ai.category.request",
    "album.ai.duplicate.request",
    "album.ai.quality.request",
    "album.ai.score.request",
    "album.ai.embedding.request",
    "album.ai.people.request",
]

MODEL_MAP = {
    "album.ai.embedding.request": embedding_schema.EmbeddingKafkaRequest,
    "album.ai.people.request": people_schema.PeopleKafkaRequest,
    "album.ai.category.request": categories_schema.CategoriesKafkaRequest,
    "album.ai.duplicate.request": duplicate_schema.DuplicateKafkaRequest,
    "album.ai.quality.request": quality_schema.QualityKafkaRequest,
    "album.ai.score.request": score_schema.ScoreKafkaRequest,
}

HANDLER_MAP = {
    "album.ai.embedding.request": embedding_handler.handle,
    "album.ai.people.request": people_handler.handle,
    "album.ai.category.request": category_handler.handle,
    "album.ai.duplicate.request": duplicate_handler.handle,
    "album.ai.quality.request": quality_handler.handle,
    "album.ai.score.request": score_handler.handle,
}

async def run_kafka_consumer(topic: str, group_id: str):
    consumer = create_kafka_consumer([topic], group_id, KAFKA_BROKER_URL)
    producer = create_kafka_producer(KAFKA_BROKER_URL)

    await consumer.start()
    await producer.start()

    print(f"[Kafka] 컨슈머, 프로듀서 연결 성공 - 컨슈머 그룹: {group_id}")

    try:
        while True:
            messages = await consumer.getmany(timeout_ms=200)
            
            if messages:
                print(f"[Kafka] getmany 결과: {[(tp.topic, len(batch)) for tp, batch in messages.items()]}", flush=True)

            tasks = [
                process_partition_batch(tp, batch, producer, HANDLER_MAP, MODEL_MAP, group_id)
                for tp, batch in messages.items() if batch
            ]
            await asyncio.gather(*tasks)
    finally:
        await consumer.stop()
        await producer.stop()
