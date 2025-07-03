import json
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

# 핸들러 (처리 함수)
from app.kafka.handler import (
    embedding as embedding_handler,
    duplicate as duplicate_handler,
    quality as quality_handler,
    category as category_handler,
    score as score_handler,
    people as people_handler,
)

# Pydantic 모델
from app.schemas.kafka import (
    embedding as embedding_schema,
    duplicate as duplicate_schema,
    quality as quality_schema,
    categories as categories_schema,
    score as score_schema,
    people as people_schema,
)

from app.config.kafka_config import (
    KAFKA_REQUEST_TOPICS, KAFKA_GROUP_ID, KAFKA_RESPONSE_TOPIC_MAP
)

TOPIC_MODEL_MAP = {
    "album.ai.embedding.request": embedding_schema.EmbeddingKafkaRequest,
    "album.ai.category.request": categories_schema.CategoriesKafkaRequest,
    "album.ai.people.request": people_schema.PeopleKafkaRequest,
    "album.ai.duplicate.request": duplicate_schema.DuplicateKafkaRequest,
    "album.ai.quality.request": quality_schema.QualityKafkaRequest,
    "album.ai.score.request": score_schema.ScoreKafkaRequest,
}

ALL_HANDLERS = {
    "album.ai.embedding.request": embedding_handler.handle,
    "album.ai.category.request": category_handler.handle,
    "album.ai.people.request": people_handler.handle,
    "album.ai.duplicate.request": duplicate_handler.handle,
    "album.ai.quality.request": quality_handler.handle,
    "album.ai.score.request": score_handler.handle,
}

def create_kafka_consumer(bootstrap_servers: str) -> AIOKafkaConsumer:
    return AIOKafkaConsumer(
        *KAFKA_REQUEST_TOPICS,
        bootstrap_servers=bootstrap_servers,
        group_id=KAFKA_GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        isolation_level="read_committed",
        max_poll_records=100
    )


async def process_partition_batch(tp, batch, producer: AIOKafkaProducer):
    topic = tp.topic
    handler = ALL_HANDLERS.get(topic)
    model_cls = TOPIC_MODEL_MAP.get(topic)
    if not handler or not model_cls:
        print("핸들러 또는 모델 없음")
        return

    data_batch = [model_cls(**msg.value) for msg in batch]
    keys = [msg.key.decode() if msg.key else None for msg in batch]
    response_topic = KAFKA_RESPONSE_TOPIC_MAP[topic]

    txn_started = False

    try:
        result_list = await handler(data_batch)

        await producer.begin_transaction()
        txn_started = True

        print("트랜잭션 시작")

        for result, key in zip(result_list, keys):
            await producer.send(
                topic=response_topic,
                key=key.encode() if key else None,
                value=result.dict()
            )
            print(f"발행 완료: taskId={result.taskId}")

        offsets = {tp: batch[-1].offset + 1}
        await producer.send_offsets_to_transaction(offsets, KAFKA_GROUP_ID)
        await producer.commit_transaction()

    except Exception as e:
        if txn_started:
            await producer.abort_transaction()
            print(f"❌ Transaction aborted: {e}")
        else:
            print(f"핸들러 처리 중 예외 발생 (트랜잭션 시작 전): {e}")

async def consume_loop(consumer: AIOKafkaConsumer, producer: AIOKafkaProducer):
    while True:
        messages = await consumer.getmany(timeout_ms=200)

        tasks = []
        for tp, batch in messages.items():
            if batch:
                tasks.append(process_partition_batch(tp, batch, producer))

        await asyncio.gather(*tasks)