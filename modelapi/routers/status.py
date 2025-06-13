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
