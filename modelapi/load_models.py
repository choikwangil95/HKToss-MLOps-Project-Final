from tokenizers import Tokenizer
from pathlib import Path
import onnxruntime as ort
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
import numpy as np
import joblib


def get_summarize_model():
    """
    ONNX 요약 모델과 토크나이저 로딩
    """
    base_path = Path("models/kobart_summary_onnx")

    encoder_sess = ort.InferenceSession(str(base_path / "encoder_model.onnx"))
    decoder_sess = ort.InferenceSession(str(base_path / "decoder_model.onnx"))
    tokenizer = Tokenizer.from_file(str(base_path / "tokenizer.json"))

    return encoder_sess, decoder_sess, tokenizer


def get_ner_tokenizer():
    """
    ONNX NER 모델과 토크나이저 로딩
    """
    base_path = Path("models/ner_onnx")

    tokenizer = Tokenizer.from_file(str(base_path / "tokenizer.json"))
    session = ort.InferenceSession(str(base_path / "model.onnx"))

    return tokenizer, session


def get_embedding_tokenizer():
    """
    ONNX NER 모델과 토크나이저 로딩
    """
    base_path = Path("models/kr_sbert_mean_onnx")

    tokenizer = Tokenizer.from_file(str(base_path / "tokenizer.json"))
    session = ort.InferenceSession(str(base_path / "kr_sbert.onnx"))

    return tokenizer, session


def get_vectordb():
    """
    vectordb 로딩
    """

    class OnnxEmbedder(Embeddings):
        def __init__(self, model_path: str, tokenizer_path: str):
            self.session = ort.InferenceSession(model_path)
            self.tokenizer = Tokenizer.from_file(tokenizer_path)

        def _embed(self, text: str):
            encoding = self.tokenizer.encode(text)
            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([[1] * len(encoding.ids)], dtype=np.int64)

            outputs = self.session.run(
                None, {"input_ids": input_ids, "attention_mask": attention_mask}
            )

            raw_vector = outputs[0][0]
            norm_vector = raw_vector / (np.linalg.norm(raw_vector) + 1e-10)
            return norm_vector.tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._embed(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._embed(text)

    model_base_path = Path("models")

    embedding = OnnxEmbedder(
        model_path=str(model_base_path / "kr_sbert_mean_onnx/kr_sbert.onnx"),
        tokenizer_path=str(model_base_path / "kr_sbert_mean_onnx/tokenizer.json"),
    )

    db_base_path = Path("db")

    vectordb = Chroma(
        persist_directory=str(db_base_path / "chroma_store"),
        embedding_function=embedding,
    )

    return vectordb


def get_lda_model():
    """
    LDA 모델과 토크나이저 로딩
    """
    model_base_path = Path("models")

    count_vectorizer = joblib.load(str(model_base_path / "count_vectorizer.pkl"))
    lda_model = joblib.load(str(model_base_path / "best_lda_model.pkl"))

    db_base_path = Path("db")

    with open(str(db_base_path / "stopwords-ko.txt"), "r", encoding="utf-8") as f:
        stopwords = [word.strip() for word in f.readlines()]

    return lda_model, count_vectorizer, stopwords

def get_prediction_models():
    """
    예측 모델과 스케일러 로딩
    """
    model_base_path = Path("models/saved_models")

    # ONNX 추론 세션 생성
    sess = ort.InferenceSession(
        str(model_base_path / "predictor.onnx"),
        providers=['CPUExecutionProvider']
    )
    
    # 타겟 스케일러 로드
    target_scaler = joblib.load(str(model_base_path / "target_scaler.joblib"))
    
    # 그룹별 스케일러 동적 로딩
    fitted_scalers = {
        i: joblib.load(str(model_base_path / f"scaler_group_{i}.joblib"))
        for i in range(9)
    }

    return sess, target_scaler, fitted_scalers
