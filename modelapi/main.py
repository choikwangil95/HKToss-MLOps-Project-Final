from fastapi import FastAPI
from routers import status, model, custom
from load_models import (
    NewsTossChatbot,
    get_embedding_tokenizer,
    get_lda_model,
    get_summarize_model,
    get_ner_tokenizer,
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
    logger.info("🟡 summarize 모델 불러오는 중...")

    encoder_sess_summarize, decoder_sess_summarize, tokenizer_summarize = (
        get_summarize_model()
    )
    app.state.encoder_sess_summarize = encoder_sess_summarize
    app.state.decoder_sess_summarize = decoder_sess_summarize
    app.state.tokenizer_summarize = tokenizer_summarize

    logger.info("🟢 summarize 모델 로딩 완료")

    logger.info("🟡 NER 모델 불러오는 중...")

    tokenizer_ner, session_ner = get_ner_tokenizer()
    app.state.tokenizer_ner = tokenizer_ner
    app.state.session_ner = session_ner

    logger.info("🟢 NER 모델 로딩 완료")

    logger.info("🟡 embedding 모델 불러오는 중...")

    tokenizer_embedding, session_embedding = get_embedding_tokenizer()
    app.state.tokenizer_embedding = tokenizer_embedding
    app.state.session_embedding = session_embedding

    logger.info("🟢 embedding 모델 로딩 완료")

    logger.info("🟡 vectordb 불러오는 중...")

    vectordb = get_vectordb()
    app.state.vectordb = vectordb

    logger.info("🟢 vectordb 모델 로딩 완료")

    logger.info("🟡 LDA 불러오는 중...")

    lda_model, count_vectorizer, stopwords = get_lda_model()
    app.state.lda_model = lda_model
    app.state.count_vectorizer = count_vectorizer
    app.state.stopwords = stopwords

    logger.info("🟢 LDA 모델 로딩 완료")

    logger.info("🟡 LLM 모델 불러오는 중...")

    chatbot = NewsTossChatbot()
    app.state.chatbot = chatbot

    logger.info("🟢 LLM 모델 로딩 완료")


# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(model.router)
app.include_router(custom.router)
