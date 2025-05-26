from fastapi import FastAPI
from routers import news, status

app = FastAPI(title="MLOps API Server", version="0.0.0")


# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(news.router)  # 뉴스 관련 라우터