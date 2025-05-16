from fastapi import FastAPI
from routers import news, status
from core.init_db import init_db

app = FastAPI(title="MLOps API Server", version="1.0.0")


# 데이터베이스 초기화
@app.on_event("startup")
def on_startup():
    init_db()


# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(news.router)  # 뉴스 관련 라우터
