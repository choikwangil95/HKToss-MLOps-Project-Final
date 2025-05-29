from fastapi import FastAPI
<<<<<<< HEAD
from routers import news, status, ner
=======
from routers import news, status, stock
>>>>>>> develop

app = FastAPI(title="MLOps API Server", version="0.0.0")


# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(news.router)  # 뉴스 관련 라우터
<<<<<<< HEAD
app.include_router(ner.router) # ner 관련 라우터
=======
app.include_router(stock.router)  # 주식 관련 라우터
>>>>>>> develop
