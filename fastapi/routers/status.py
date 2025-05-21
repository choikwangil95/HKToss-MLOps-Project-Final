from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


@router.get(
    "/healthcheck",
    response_model=HealthResponse,
    tags=["System"],
    summary="헬스 체크",
    description="서버 상태를 확인하는 헬스체크 엔드포인트입니다.",
)
def healthcheck():
    """
    서버가 정상적으로 작동 중인지 확인하는 API입니다.
    """
    return {"status": "ok"}


@router.get(
    "/ping",
    response_class=PlainTextResponse,
    tags=["System"],
    summary="핑 체크",
    description="단순한 핑/퐁 응답으로 서버 응답 여부 확인 (텍스트 응답)",
)
def ping():
    """
    서버가 응답 가능한지 빠르게 확인할 수 있는 Ping 테스트입니다.
    """
    return "pong"
