import asyncio
from functools import wraps

from loguru import logger


def log_exception(func):
    """예외 발생 시 자동으로 로깅하는 데코레이터"""

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.opt(depth=1).exception(
                f"{func.__name__} 함수 예외 발생: {e}"
            )
            raise

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.opt(depth=1).exception(
                f"{func.__name__} 함수 예외 발생: {e}"
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def log_flow(func):
    """함수 플로우 로깅 데코레이터"""

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger.opt(depth=2).info(f"{func.__name__} 함수 시작")

        try:
            result = func(*args, **kwargs)
            logger.opt(depth=2).info(f"{func.__name__} 함수 성공")
            return result

        except Exception as e:
            logger.opt(depth=2).exception(
                f"{func.__name__} 함수 예외 발생: {e}"
            )
            raise

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger.opt(depth=2).info(f"{func.__name__} 함수 시작")

        try:
            result = await func(*args, **kwargs)
            logger.opt(depth=2).info(f"{func.__name__} 함수 성공")
            return result

        except Exception as e:
            logger.opt(depth=2).exception(
                f"{func.__name__} 함수 예외 발생: {e}"
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
