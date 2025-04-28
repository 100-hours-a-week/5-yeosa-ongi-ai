from cachetools import TTLCache
from typing import Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# 전역 TTL 캐시 인스턴스
embedding_cache = TTLCache(maxsize=500, ttl=300)

def get_cached_embedding(key: str) -> Optional[Any]:
    """캐시에서 안전하게 값 가져오기"""
    return embedding_cache.get(key, None)

def set_cached_embedding(key: str, value: Any) -> None:
    """캐시에 안전하게 값 저장"""
    embedding_cache[key] = value

def del_embedding_cache(key: str) -> None:
    """캐시 삭제"""
    embedding_cache.pop(key, None)

def clear_embedding_cache() -> None:
    """전체 캐시 비우기 (필요 시)"""
    embedding_cache.clear()
    
def get_cached_embeddings_parallel(keys: List[str], max_workers: int = 8) -> Tuple[List[Optional[Any]], List[str]]:
    """
    병렬로 캐시에서 임베딩을 가져옴
    Args:
        keys: 캐시 key 리스트
        max_workers: 병렬 스레드 수

    Returns:
        - List of cached values (index는 keys와 동일)
        - List of keys not found in cache
    """
    def check(key: str) -> Tuple[str, Optional[Any]]:
        return key, get_cached_embedding(key)

    results = [None] * len(keys)
    missing_keys = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, (key, value) in enumerate(executor.map(check, keys)):
            results[i] = value
            if value is None:
                missing_keys.append(keys[i])
    return results, missing_keys