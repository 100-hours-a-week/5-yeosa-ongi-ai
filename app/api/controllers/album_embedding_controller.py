import time
import pickle
import asyncio
import logging
from datetime import datetime

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi import HTTPException

from app.schemas.album_schema import ImageRequest
from app.utils.logging_decorator import log_flow
from app.core.cache import set_cached_embedding

DEFAULT_BATCH_SIZE = 16
logger = logging.getLogger(__name__)

def format_elapsed(t: float) -> str:
    return f"{t * 1000:.2f} ms" if t < 1 else f"{t:.2f} s"

def now_str() -> str:
    """사람이 읽기 좋은 시각 문자열 반환"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

@log_flow
async def embed_controller(req: ImageRequest, request: Request) -> Response:
    """
    클라이언트로부터 이미지 파일명을 받아 GPU 서버에 전달하고,
    임베딩 결과를 받아 캐싱하는 컨트롤러입니다.
    """

    try:
        print('[INFO] gpu_client 불러오기')
        gpu_client = request.app.state.gpu_client
        print(f'[INFO] gpu_client 불러오기 성공! {gpu_client.base_url}')
        if req.images:
            first_image = req.images[0]
            print(f"[CHECK] GPU에 요청된 첫 번째 이미지 파일명: {first_image}")
        else:
            print("[WARN] 요청 이미지 리스트가 비어 있습니다.")

        # ✅ 전송 시작 시각
        send_time_str = now_str()
        t1 = time.time()
        print(f"[INFO] GPU 서버 전송 시작 시각: {send_time_str}")
        response = await gpu_client.post(
            "/clip/embedding",
            json=req.dict(),  # {"images": [...]}
            headers={"Content-Type": "application/json"},
        )
        t2 = time.time()
        recv_time_str = now_str()
        print(f"[INFO] GPU 서버 응답 수신 시각: {recv_time_str}")
        print(f"[INFO] GPU 요청-응답 소요 시간: {format_elapsed(t2 - t1)}")

        if response.status_code != 200:
            print(f"[ERROR] GPU 서버 응답 실패 - Status: {response.status_code}")
            return JSONResponse(
                status_code=500,
                content={"message": "embedding failed (gpu error)", "data": None}
            )

        # ✅ 응답 역직렬화 시간 측정
        t3 = time.time()
        response_obj = pickle.loads(await response.aread())  # bytes → object
        t4 = time.time()

        if response_obj.get("message") != "success":
            print(f"[ERROR] GPU 서버 응답 비정상 - message: {response_obj.get('message')}")
            return JSONResponse(
                status_code=500,
                content={"message": "embedding failed (gpu processing error)", "data": None}
            )

        result = response_obj["data"]  # Dict[str, List[float]]

        first_filename = next(iter(result.keys()), None)
        if first_filename:
            print(f"[CHECK] GPU 응답 첫 번째 이미지 파일명: {first_filename}")
        else:
            print("[WARN] 결과에서 파일명이 하나도 없습니다.")


        for filename, feature in result.items():
            try:
                await set_cached_embedding(filename, feature)
            except Exception as e:
                logger.error(f"[Redis SET ERROR] key='{filename}' failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "embedding succeeded, but caching failed",
                        "failed_keys": [{"key": filename, "error": str(e)}],
                    }
                )

        return JSONResponse(status_code=201, content={"message": "success", "data": None})

    except Exception as e:
        logger.exception("[EXCEPTION] embed_controller 처리 중 오류 발생")
        return JSONResponse(
            status_code=500,
            content={"message": "embedding failed (exception)", "data": None}
        )
