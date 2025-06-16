from tokenizers import Tokenizer
from pathlib import Path
import onnxruntime as ort
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
import numpy as np
import joblib
import os
import requests
import openai
from dotenv import load_dotenv

load_dotenv()


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


class NewsTossChatbot:
    def __init__(self):
        # OpenAI 클라이언트 준비 (환경변수 OPENAI_API_KEY 필요)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_client(self):
        return self.client

    def search_similar_news(self, query_text, top_k=5):
        url = "http://15.165.211.100:8000/news/similar"
        payload = {"article": query_text, "top_k": top_k}

        response = requests.post(url, json=payload)
        response.raise_for_status()
        similar_news = response.json()["similar_news_list"]

        return similar_news

    def build_prompt(self, context, question):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        당신은 주식 투자자를 위한 뉴스 기반 정보 어시스턴트 챗봇 '뉴스토스'입니다.
        당신의 임무는 실시간 뉴스와 과거 유사사건 뉴스 데이터를 바탕으로,
        - 사용자의 투자 판단에 도움이 되는 정보를 제공하고,
        - 뉴스에서 과거 유사사건, 해당 시기의 주가 흐름, 관련 리포트의 핵심 내용을 구체적으로 찾아 인용하며,
        - 미래 전망 질문에는 과거 사례를 근거로 신중하게 의견을 제시하는 것입니다.

        답변 작성 시 반드시 다음을 지켜주세요:
        1. 답변 내용 중 포함되는 과거 유사사건의 날짜, 사건명, 당시 주가 흐름(상승/하락/횡보 등), 주요 리포트 내용은 구체적으로 인용하세요.
        2. 미래 전망 질문에는 과거 유사사건을 근거로 논리적인 전망을 제시하세요.
        3. 답변 마지막에는 '⭐️투자 결과에 대한 책임은 본인에게 있습니다.⭐️'라는 안내문을 추가하세요.
        4. 답변은 반드시 한글로, 명확하고 간결하게 작성하세요.
        5. 제공된 검색 결과(유사도 높은 과거 뉴스, 주가 데이터, 리포트 등)만 근거로 사용하세요. 근거가 없으면 '근거가 없는데 답변해도 될까? 이건 너의 소중한 돈이 걸린 문제야 ^^;;'라고 하세요.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        검색 결과: {context}
        질문: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def make_stream_prompt(self, question, top_k=5):
        # top_k 뉴스 검색 및 context 생성
        similar_news = self.search_similar_news(question, top_k=top_k)
        retrieved_infos = []
        for similar_news_item in similar_news:
            row = similar_news_item
            info = (
                # f"[{row['wdate']}] {row['press']} - {row['title']}\n"
                # f"URL: {row['url']}\n"
                f"[{row['wdate']}]\n"
                f"{row['summary']}\n"
                f"(코사인유사도: {row['score']:.4f})"
            )
            retrieved_infos.append(info)
        context = "\n\n".join(retrieved_infos)
        return self.build_prompt(context, question)

    def answer(self, question, top_k=5):
        prompt = self.make_stream_prompt(question, top_k)

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
        )

        return response.choices[0].message.content
