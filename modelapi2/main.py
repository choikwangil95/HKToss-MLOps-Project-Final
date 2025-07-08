from fastapi import FastAPI
from routers import status, custom
from load_models import (
    NewsTossChatbot,
    get_embedding_tokenizer,
    get_recommend_ranker_model,
    get_recommend_model,
    get_vectordb,
)
from monitoring import instrumentator
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MLOps Model API Server", version="0.0.0")

# 모니터링
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)


@app.on_event("startup")
async def startup_event():
    logger.info("🟡 embedding 모델 불러오는 중...")

    tokenizer_embedding, session_embedding = get_embedding_tokenizer()
    app.state.tokenizer_embedding = tokenizer_embedding
    app.state.session_embedding = session_embedding

    logger.info("🟢 embedding 모델 로딩 완료")

    logger.info("🟡 vectordb 불러오는 중...")

    vectordb = get_vectordb()
    app.state.vectordb = vectordb

    logger.info("🟢 vectordb 모델 로딩 완료")

    logger.info("🟡 LLM 모델 불러오는 중...")

    chatbot = NewsTossChatbot()
    app.state.chatbot = chatbot

    logger.info("🟢 LLM 모델 로딩 완료")

    logger.info("🟡 뉴스 추천 모델 불러오는 중...")

    model_recommend = get_recommend_model()
    app.state.model_recommend = model_recommend

    logger.info("🟢 뉴스 추천 모델 로딩 완료")

    logger.info("🟡 뉴스 추천 랭킹 모델 불러오는 중...")

    model_recommend_ranker = get_recommend_ranker_model()
    app.state.model_recommend_ranker = model_recommend_ranker

    logger.info("🟢 뉴스 추천 랭킹 모델 로딩 완료")


# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(custom.router)
