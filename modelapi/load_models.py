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
import onnxruntime as ort

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

    def search_similar_news(self, query_text, top_k=10):
        url = "http://15.165.211.100:8000/news/similar"
        payload = {"article": query_text, "top_k": top_k}

        response = requests.post(url, json=payload)
        response.raise_for_status()
        similar_news = response.json()["similar_news_list"]

        return similar_news

    def build_prompt(self, context, question, has_news=True):
        if has_news:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            당신이 누구냐 묻는다면, "저는 과거 뉴스 정보를 바탕으로 답변해드리는 챗봇 뉴스토스입니다.😄" 라고 답하세요.
            어떠한 질문이든 반드시 아래 예시 포맷만 출력하세요. 추가 의견, 종합 정보 등은 절대 포함하지 마세요.

            [답변 작성 시 반드시 다음을 지켜주세요]
            1. 답변에는 반드시 과거 유사사건 뉴스 정보를 아래와 같은 카드 형태로 정리해 보여주세요:
                - 날짜, 제목(하이퍼링크), 언론사, 요약, 관련 이미지(아래 예시 참고)
                - 예시:
                ■ [2024-11-28] "SK하이닉스, 신규 주주환원책으로 재무구조 개선 기대"
                (https://n.news.naver.com/mnews/article/008/0005120417)
                ▶ 언론사: 머니투데이
                ▶ 유사도: 0.56
                ▶ 요약: NH투자증권이 신규 주주환원 정책을 공시한 SK하이닉스에 대해...
                ▶ 관련 이미지: <img src="https://imgnews.pstatic.net/image/008/2024/11/28/0005120417_001_20241128085813446.jpg?type=w800" alt="뉴스 이미지">
            2. 유사 사건 뉴스 정보 외 다른 의견, 종합 안내 정보 등은 절대로 제시하지 마세요. 

            [제공된 유사 뉴스 카드]
            {context}

            [사용자 질문]
            {question}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        else:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            유사 뉴스가 없다면, 절대로 의견이나 종합 안내 정보 등을 제시하지 말고 아래 3가지 답변 예시 중 하나로만 답변하세요:
            - "현재 제공된 뉴스 카드 중에서는 이번 질문과 직접적으로 연결된 사례는 확인되지 않지만, 뉴스토스는 항상 최신 이슈와 다양한 데이터를 바탕으로 최선을 다해 안내해드리고 있습니다. 궁금하신 점이나 더 구체적인 관심 분야가 있다면 언제든 말씀해 주세요!"
            - "질문하신 내용과 가장 가까운 사례를 찾기 위해 노력했지만, 이번에는 제공된 뉴스 카드 내에서 직접적인 연관 사례를 확인하기 어려웠습니다. 앞으로도 더 정확하고 풍부한 정보를 드릴 수 있도록 계속 업데이트하고 있으니, 궁금한 점이 있으시면 언제든 질문해 주세요!"
            - "더 구체적으로 질문해주시면, 정확한 답변을 드릴 수 있습니다!"

            [사용자 질문]
            {question}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


    def make_stream_prompt(self, question, top_k=10):
        similar_news = self.search_similar_news(question, top_k=top_k)
        # 0.1 이상만 필터링
        filtered_news = [row for row in similar_news if row.get('similarity', 0) >= 0.1]
        retrieved_infos = []
        for row in filtered_news:
            info = (
                f"{row['title']} ({row['url']})\n"
                f"<img src=\"{row['image']}\" alt=\"뉴스 이미지\">\n"
                f"{row['summary']}\n"
                f"{row['wdate'][:10]} {row.get('press', '정보없음')}\n"
                f"(유사도: {row.get('similarity', 0):.2f})"
            )
            retrieved_infos.append(info)
        context = "\n\n".join(retrieved_infos)
        return self.build_prompt(context, question, has_news=bool(filtered_news))


    def answer(self, question, top_k=10):
        prompt = self.make_stream_prompt(question, top_k)

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
        )

        return response.choices[0].message.content


def get_recommend_model():
    model_base_path = Path("models")

    # ONNX 세션 생성
    model_recommend = ort.InferenceSession(
        str(model_base_path / "two_tower_model.onnx")
    )

    return model_recommend
