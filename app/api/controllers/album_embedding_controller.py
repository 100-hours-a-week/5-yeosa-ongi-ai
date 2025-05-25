import time
import pickle
from datetime import datetime

from fastapi import Request
from fastapi.responses import JSONResponse

from app.schemas.album_schema import ImageRequest
from app.utils.logging_decorator import log_flow
from app.core.cache import set_cached_embedding

DEFAULT_BATCH_SIZE = 16

def format_elapsed(t: float) -> str:
    return f"{t * 1000:.2f} ms" if t < 1 else f"{t:.2f} s"

def now_str() -> str:
    """사람이 읽기 좋은 시각 문자열 반환"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

@log_flow
async def embed_controller(req: ImageRequest, request: Request) -> JSONResponse:
    """
    클라이언트로부터 이미지 파일명을 받아 GPU 서버에 전달하고,
    임베딩 결과를 받아 캐싱하는 컨트롤러입니다.
    """
    print(f"[START] 이미지 임베딩 요청 시작 - 총 이미지 수: {len(req.images)}")

    try:
        gpu_client = request.app.state.gpu_client

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
        print(f"[INFO] 요청-응답 소요 시간: {format_elapsed(t2 - t1)}")

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
        print(f"[INFO] 응답 역직렬화 완료: {format_elapsed(t4 - t3)}")

        if response_obj.get("message") != "success":
            print(f"[ERROR] GPU 서버 응답 비정상 - message: {response_obj.get('message')}")
            return JSONResponse(
                status_code=500,
                content={"message": "embedding failed (gpu processing error)", "data": None}
            )

        result = response_obj["data"]  # Dict[str, List[float]]
        print(f"[INFO] 임베딩 완료 - 처리된 이미지 수: {len(result)}")

        for filename, feature in result.items():
            set_cached_embedding(filename, feature)

        print("[SUCCESS] 임베딩 결과 캐싱 완료")
        return JSONResponse(status_code=201, content={"message": "success", "data": None})

    except Exception as e:
        print(f"[EXCEPTION] GPU 서버 호출 중 오류 발생: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "embedding failed (exception)", "data": None}
        )
