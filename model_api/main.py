from fastapi import FastAPI
from routers import news, status
import os
import pickle
from sentence_transformers import SentenceTransformer

app = FastAPI(title="MLOps API Server", version="0.0.0")

# 모델 추가
MODEL_DIR = "./ml_models"
MODEL_PATH = os.path.join(MODEL_DIR, "ko-sbert-sts_model.pk")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("모델 다운로드 중...", flush=True)  # 바로 출력
    model = SentenceTransformer("jhgan/ko-sbert-sts")
    model.to("cpu")  # CPU 강제 지정 (핵심)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"모델 저장 완료: {MODEL_PATH}")
else:
    print(f"이미 저장된 모델이 존재합니다: {MODEL_PATH}")


# 라우터
app.include_router(status.router)  # 상태 확인 및 헬스 체크 라우터
app.include_router(news.router)  # 뉴스 관련 라우터
