from fastapi import FastAPI
from routers import status, model, custom
from load_models import (
    get_embedding_tokenizer,
    get_lda_model,
    get_summarize_model,
    get_ner_tokenizer,
    get_vectordb,
    get_prediction_models,
)
from monitoring import instrumentator

app = FastAPI(title="MLOps Model API Server", version="0.0.0")

# 모니터링
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)


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

    print("🟡 LDA 불러오는 중...")

    lda_model, count_vectorizer, stopwords = get_lda_model()
    app.state.lda_model = lda_model
    app.state.count_vectorizer = count_vectorizer
    app.state.stopwords = stopwords
    
    print("🟢 LDA 모델 로딩 완료")

    print("🟡 예측 모델 불러오는 중...")

    predictor, target_scaler, group_scalers = get_prediction_models()
    app.state.predictor = predictor
    app.state.target_scaler = target_scaler
    app.state.group_scalers = group_scalers

    print("🟢 예측 모델 로딩 완료")




# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(model.router)
app.include_router(custom.router)
