import os
from dotenv import load_dotenv

from redis.asyncio import Redis, ConnectionPool
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")

# 예외 처리
if not REDIS_HOST:
    raise EnvironmentError("REDIS_HOST가 .env 파일에 없습니다.")
if not REDIS_PORT:
    raise EnvironmentError("REDIS_PORT가 .env 파일에 없습니다.")
if not REDIS_DB:
    raise EnvironmentError("REDIS_DB가 .env 파일에 없습니다.")

_redis: Redis | None = None

def init_redis() -> Redis:
    global _redis
    pool = ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        max_connections=100,
        decode_responses=True,
    )

    retry_strategy = Retry(
        backoff=ExponentialBackoff(base=1, cap=10), 
        retries=5
    )

    _redis = Redis(connection_pool=pool, 
                   retry=retry_strategy, 
                   retry_on_error=[ConnectionError, TimeoutError])
    return _redis

def get_redis() -> Redis:
    if _redis is None:
        raise RuntimeError("Redis가 초기화되지 않았습니다.")
    return _redis
