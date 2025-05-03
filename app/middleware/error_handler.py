from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

def setup_exception_handler(app: FastAPI):
    @app.middleware("http")
    async def catch_exceptions_middleware(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.exception("middleware error")
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})