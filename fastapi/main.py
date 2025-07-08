from fastapi import FastAPI
from routers import news, status, user
from monitoring import instrumentator


app = FastAPI(title="MLOps API Server", version="0.0.0")


# 모니터링
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)


# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(news.router)  # 뉴스 관련 라우터
app.include_router(news.router_v2)  # 뉴스 관련 라우터
app.include_router(user.router)  # 뉴스 관련 라우터
