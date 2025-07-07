import json
from typing import Callable, Any
from aiokafka import AIOKafkaConsumer
from app.config.kafka_config import KAFKA_RESPONSE_TOPIC_MAP

def create_kafka_consumer(topics: list[str], group_id: str, bootstrap_servers: str) -> AIOKafkaConsumer:
    return AIOKafkaConsumer(
        *topics,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        isolation_level="read_committed",
        max_poll_records=100
    )

async def process_partition_batch(tp, batch, producer, handler_map: dict, model_map: dict, group_id: str):
    topic = tp.topic
    handler: Callable = handler_map.get(topic)
    model_cls: Any = model_map.get(topic)
    if not handler or not model_cls:
        print(f"[{topic}] 핸들러 또는 모델 없음")
        return

    data_batch = [model_cls(**msg.value) for msg in batch]
    keys = [msg.key.decode() if msg.key else None for msg in batch]
    response_topic = KAFKA_RESPONSE_TOPIC_MAP[topic]

    txn_started = False
    try:
        result_list = await handler(data_batch)

        await producer.begin_transaction()
        txn_started = True

        for result, key in zip(result_list, keys):
            await producer.send(
                topic=response_topic,
                key=key.encode() if key else None,
                value=result.dict()
            )

        offsets = {tp: batch[-1].offset + 1}
        await producer.send_offsets_to_transaction(offsets, group_id)
        await producer.commit_transaction()

    except Exception as e:
        if txn_started:
            await producer.abort_transaction()
        print(f"[{topic}] ❌ 트랜잭션 처리 중 에러: {e}")