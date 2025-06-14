from fastapi import FastAPI
from routers import status, model
from load_models import (
    get_embedding_tokenizer,
    get_summarize_model,
    get_ner_tokenizer,
    get_vectordb,
)

app = FastAPI(title="MLOps Model API Server", version="0.0.0")


@app.on_event("startup")
async def startup_event():
    print("🟡 summarize 모델 불러오는 중...")

    encoder_sess_summarize, decoder_sess_summarize, tokenizer_summarize = (
        get_summarize_model()
    )
    app.state.encoder_sess_summarize = encoder_sess_summarize
    app.state.decoder_sess_summarize = decoder_sess_summarize
    app.state.tokenizer_summarize = tokenizer_summarize

    print("🟢 summarize 모델 로딩 완료")

    print("🟡 NER 모델 불러오는 중...")

    tokenizer_ner, session_ner = get_ner_tokenizer()
    app.state.tokenizer_ner = tokenizer_ner
    app.state.session_ner = session_ner

    print("🟢 NER 모델 로딩 완료")

    print("🟡 embedding 모델 불러오는 중...")

    tokenizer_embedding, session_embedding = get_embedding_tokenizer()
    app.state.tokenizer_embedding = tokenizer_embedding
    app.state.session_embedding = session_embedding

    print("🟢 embedding 모델 로딩 완료")

    print("🟡 vectordb 불러오는 중...")

    vectordb = get_vectordb()
    app.state.vectordb = vectordb

    print("🟢 vectordb 모델 로딩 완료")


# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(model.router)  # 상태 확인 및 헬스 체크 라우터
