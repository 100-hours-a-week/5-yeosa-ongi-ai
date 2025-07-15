from aiokafka import AIOKafkaProducer
import json
import uuid


def create_kafka_producer(bootstrap_servers: str) -> AIOKafkaProducer:
    return AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers,
        transactional_id=f"producer-{uuid.uuid4()}",
        acks="all",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
