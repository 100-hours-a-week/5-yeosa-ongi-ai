import json, logging
from typing import Any
import traceback
import os
import asyncio
from dotenv import load_dotenv

import torch

from app.config.redis import get_redis

load_dotenv()
logger = logging.getLogger(__name__)

try:
    raw_ttl = os.getenv("REDIS_CACHE_TTL")
    if raw_ttl is None:
        raise ValueError("REDIS_CACHE_TTL이 .env 파일에 없습니다.")
    REDIS_CACHE_TTL = int(raw_ttl)
except ValueError as e:
    raise EnvironmentError(str(e))
except Exception as e:
    raise EnvironmentError("REDIS_CACHE_TTL이 정수로 지정되어야 합니다.")

async def get_cached_embedding(key: str) -> Any | None:
    from app.config.app_config import get_config
    redis = get_redis()
    semaphore = get_config().redis_semaphore

    try:
        async with semaphore:
            value = await redis.get(key)

        if value is None:
            logger.warning(f"[Redis GET] key='{key}' not found")
            return None

        return torch.tensor(json.loads(value))
    
    except Exception as e:
        logger.error(f"[Redis GET ERROR] key='{key}' failed: {e}", exc_info=True)
        return None


async def set_cached_embedding(key: str, value: Any) -> None:
    from app.config.app_config import get_config
    redis = get_redis()
    semaphore = get_config().redis_semaphore

    try:
        if isinstance(value, torch.Tensor):
            value = value.cpu().float().tolist()

        ttl = int(REDIS_CACHE_TTL)
        async with semaphore:
            result = await redis.set(key, json.dumps(value), ex=ttl)

        if not result:
            raise RuntimeError(f"Redis SET failed for key='{key}' (result=False)")

    except Exception as e:
        logger.error(f"[Redis SET ERROR] key='{key}' failed: {e}", exc_info=True)
        raise


async def del_embedding_cache(key: str) -> None:
    from app.config.app_config import get_config
    redis = get_redis()
    semaphore = get_config().redis_semaphore

    try:
        async with semaphore:
            await redis.delete(key)

    except Exception as e:
        logger.error(f"[Redis DEL ERROR] key='{key}' failed: {e}", exc_info=True)

async def clear_embedding_cache() -> None:
    from app.config.app_config import get_config
    redis = get_redis()
    semaphore = get_config().redis_semaphore

    try:
        async with semaphore:
            keys = await redis.keys("*")

        if keys:
            await redis.delete(*keys)
    
    except Exception as e:
        logger.error("[Redis CLEAR ERROR] 캐시 삭제 실패", exc_info=True)

async def get_cached_embeddings_parallel(keys: list[str]) -> tuple[list[Any | None], list[str]]:
    """
    비동기로 여러 키를 Redis에서 조회합니다. 실패한 키도 기록합니다.
    """

    results = await asyncio.gather(
        *(get_cached_embedding(key) for key in keys),
        return_exceptions=True
    )

    final_results = []
    missing_keys = []

    for key, result in zip(keys, results):
        if isinstance(result, Exception):
            missing_keys.append(key)
            logger.warning(
                f"[Redis ERROR] key='{key}' 조회 실패: {type(result).__name__}: {result}",
                extra={"traceback": traceback.format_exception_only(type(result), result)}
            )
            final_results.append(None)
        elif result is None:
            missing_keys.append(key)
            logger.info(f"[Redis MISS] key='{key}' not found in cache.")
            final_results.append(None)
        else:
            final_results.append(result)

    return final_results, missing_keys