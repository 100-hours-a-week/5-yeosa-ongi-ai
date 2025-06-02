import json
from typing import Any
import os
import asyncio
import time
from dotenv import load_dotenv

import torch
from app.config.redis import get_redis

load_dotenv()
REDIS_CACHE_TTL = os.getenv("REDIS_CACHE_TTL")

# 예외 처리
if not REDIS_CACHE_TTL:
    raise EnvironmentError("REDIS_CACHE_TTL이 .env 파일에 없습니다.")


async def get_cached_embedding(key: str) -> Any | None:
    redis = get_redis()

    # 시간 테스트
    t0 = time.perf_counter()
    value = await redis.get(key)
    t1 = time.perf_counter()
    print(f"[Redis GET] key='{key}' took {t1 - t0:.4f} seconds")

    if value is None:
        return None
    try:
        return torch.tensor(json.loads(value))
    except Exception:
        return value


async def set_cached_embedding(key: str, value: Any) -> None:
    redis = get_redis()
    if isinstance(value, torch.Tensor):
        value = value.cpu().float().tolist()

    # 시간 테스트
    t0 = time.perf_counter()
    await redis.set(key, json.dumps(value), ex=REDIS_CACHE_TTL)
    t1 = time.perf_counter()
    print(f"[Redis SET] key='{key}' took {t1 - t0:.4f} seconds")



async def del_embedding_cache(key: str) -> None:
    redis = get_redis()

    # 시간 테스트
    t0 = time.perf_counter()
    await redis.delete(key)
    t1 = time.perf_counter()
    print(f"[Redis DEL] key='{key}' took {t1 - t0:.4f} seconds")


async def clear_embedding_cache() -> None:
    redis = get_redis()
    keys = await redis.keys("*")
    if keys:
        await redis.delete(*keys)

async def get_cached_embeddings_parallel(keys: list[str]) -> tuple[list[Any | None], list[str]]:
    """
    비동기로 여러 키를 Redis에서 조회합니다.
    """
    # 시간 테스트
    t0 = time.perf_counter()
    results = await asyncio.gather(*(get_cached_embedding(key) for key in keys))
    t1 = time.perf_counter()
    print(f"[Redis BULK GET] {len(keys)} keys took {t1 - t0:.4f} seconds")

    missing_keys = [key for key, value in zip(keys, results) if value is None]
    return results, missing_keys


###########
# 전역 TTL 캐시 인스턴스
# embedding_cache = TTLCache(maxsize=500, ttl=300)


# def get_cached_embedding(key: str) -> Any | None:
#     """캐시에서 안전하게 값 가져오기"""
#     return embedding_cache.get(key, None)


# def set_cached_embedding(key: str, value: Any) -> None:
#     """
#     캐시에 안전하게 값 저장.
#     텐서일 경우 .cpu().float() 처리하여 CPU 연산 오류 방지.
#     """
#     if isinstance(value, torch.Tensor):
#         value = value.cpu().float()
#     embedding_cache[key] = value


# def del_embedding_cache(key: str) -> None:
#     """캐시 삭제"""
#     embedding_cache.pop(key, None)


# def clear_embedding_cache() -> None:
#     """전체 캐시 비우기 (필요 시)"""
#     embedding_cache.clear()


# def get_cached_embeddings_parallel(
#     keys: list[str], max_workers: int = 8
# ) -> tuple[list[Any | None], list[str]]:
#     """
#     병렬로 캐시에서 임베딩을 가져옵니다.

#     Args:
#         keys: 캐시 key 리스트
#         max_workers: 병렬 스레드 수

#     Returns:
#         - List of cached values (index는 keys와 동일)
#         - List of keys not found in cache

#     """

#     def check(key: str) -> tuple[str, Any | None]:
#         return key, get_cached_embedding(key)

#     results = [None] * len(keys)
#     missing_keys = []

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for i, (key, value) in enumerate(executor.map(check, keys)):
#             results[i] = value
#             if value is None:
#                 missing_keys.append(keys[i])
#     return results, missing_keys
