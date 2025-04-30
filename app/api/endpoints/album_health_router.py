from fastapi import APIRouter

# HACK: Health check용 임시 라우터

router = APIRouter(tags=["health"])

@router.get("/", status_code=200)
def health_check():
    return {
        "status": "ok",
        "message": "Service is healthy",
        "version": "1.0.0"
    }
