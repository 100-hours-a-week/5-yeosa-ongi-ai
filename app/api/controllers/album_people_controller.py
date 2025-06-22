import time
from functools import partial
from datetime import datetime
from typing import Any
import asyncio

from fastapi import Request
from fastapi.responses import JSONResponse

from app.schemas.album_schema import ImageRequest
from app.utils.logging_decorator import log_exception, log_flow

PEOPLE_SEMAPHORE_SIZE = 5
people_semaphore = asyncio.Semaphore(PEOPLE_SEMAPHORE_SIZE)

def format_elapsed(t: float) -> str:
    return f"{t * 1000:.2f} ms" if t < 1 else f"{t:.2f} s"

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

@log_flow
async def people_controller(req: ImageRequest, request: Request) -> JSONResponse:
    """
    클라이언트로부터 이미지 파일명을 받아 GPU 서버에 전달하고,
    클러스터링 결과를 받아 응답합니다.

    Args:
        req (ImageRequest): 이미지 파일 이름 리스트를 포함한 요청 객체입니다.
        request (Request): FastAPI 요청 객체. 모델 등 앱 상태에 접근하는 데 사용됩니다.

    Returns:
        dict: 클러스터링 결과를 포함한 응답. 예시:
            {
                "message": "success",
                "data": [
                    ["img001.jpg", "img005.jpg"],  # 동일 인물 A
                    ["img002.jpg"],                # 동일 인물 B
                    ...
                ]
            }

    """
    try:
        gpu_client = request.app.state.gpu_client

        # ✅ 전송 시작 시각
        send_time_str = now_str()
        t1 = time.time()
        print(f"[INFO] GPU 서버 전송 시작 시각: {send_time_str}")

        response = await gpu_client.post(
            "/people/cluster",  # GPU 서버 people 클러스터링 API 엔드포인트
            json=req.dict(),  # {"images": [...]}
            headers={"Content-Type": "application/json"},
        )


        t2 = time.time()
        recv_time_str = now_str()
        print(f"[INFO] GPU 서버 응답 수신 시각: {recv_time_str}")
        print(f"[INFO] 요청-응답 소요 시간: {format_elapsed(t2 - t1)}")

        if response.status_code != 200:
            print(f"[ERROR] GPU 서버 응답 실패 - Status: {response.status_code}")
            return JSONResponse(
                status_code=500,
                content={"message": "people clustering failed (gpu error)", "data": None}
            )

        response_obj = response.json() 

        if response_obj.get("message") != "success":
            print(f"[ERROR] GPU 서버 응답 비정상 - message: {response_obj.get('message')}")
            return JSONResponse(
                status_code=500,
                content={"message": "people clustering failed (gpu processing error)", "data": None}
            )

        result = response_obj["data"]
        print(f"[SUCCESS] 인물 클러스터링 완료 - 군집 수: {len(result)}")

        return JSONResponse(status_code=201, content={"message": "success", "data": result})

    except Exception as e:
        print(f"[EXCEPTION] GPU 서버 호출 중 오류 발생: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "people clustering failed (exception)", "data": None}
        )