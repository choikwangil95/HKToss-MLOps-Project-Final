from fastapi import FastAPI
from routers import news, status, stock, ner  

app = FastAPI(title="MLOps API Server", version="0.0.0")

# 라우터 등록
app.include_router(status.router)  # 상태 확인 및 헬스 체크
app.include_router(news.router)    # 뉴스 관련
app.include_router(stock.router)   # 주식 관련
app.include_router(ner.router)     # NER 관련
