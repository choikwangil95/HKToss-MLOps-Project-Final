from fastapi import APIRouter

router = APIRouter()


@router.get("/healthcheck", tags=["System"])
def healthcheck():
    return {"status": "ok"}


@router.get("/ping", tags=["System"])
def ping():
    return "pong"
