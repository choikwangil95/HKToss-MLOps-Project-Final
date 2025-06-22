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
import pickle
import re

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


def get_prediction_models():
    """
    예측 모델과 스케일러 로딩
    """
    model_base_path = Path("models/saved_models")

    # ONNX 추론 세션 생성
    sess = ort.InferenceSession(
        str(model_base_path / "predictor.onnx"), providers=["CPUExecutionProvider"]
    )

    # 타겟 스케일러 로드
    target_scaler = joblib.load(str(model_base_path / "target_scaler.joblib"))

    # 그룹별 스케일러 동적 로딩
    fitted_scalers = {
        i: joblib.load(str(model_base_path / f"scaler_group_{i}.joblib"))
        for i in range(9)
    }

    return sess, target_scaler, fitted_scalers


class NewsTossChatbot:
    def __init__(self):
        # OpenAI 클라이언트 준비 (환경변수 OPENAI_API_KEY 필요)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_client(self):
        return self.client

    def search_similar_news(self, query_text, top_k=2):
        # Step 1: 첫 번째 API로 query_text 기반 가장 유사한 뉴스 1개 찾기
        first_url = "http://15.165.211.100:8000/news/similar"
        response = requests.post(first_url, json={"article": query_text, "top_k": 1})
        response.raise_for_status()
        top_news = response.json()["similar_news_list"][0]
        news_id = top_news["news_id"]

        # Step 2: 해당 news_id를 두 번째 API에 넣어서 유사 뉴스 top_k개 가져오기
        second_url = f"http://3.37.207.16:8000/news/v2/{news_id}/similar?top_n=2"
        response = requests.get(second_url)
        response.raise_for_status()
        similar_news = response.json()

        return similar_news

    def build_prompt(self, context, question, has_news=True):
        if has_news:
            system_prompt = """
                당신은 과거 뉴스 정보를 바탕으로만 답변하는 전문 AI 챗봇, '뉴스토스'입니다.  
                답변은 반드시 아래 기준에 따라 작성되어야 하며, 다음 사항을 철저히 준수해야 합니다.

                ---

                **절대 포함하지 말아야 할 내용:**

                - 종합 정보  
                - 배경 설명  
                - 개인 의견  
                - 미래 예측  
                - 추가 해석  

                ---

                ## [질문 분류 기준]

                ### 1. 정체성 관련 질문
                - 예시: "너 누구야", "정체가 뭐야", "니 역할은 뭐야", "너 뭐하는 애야"
                - **이 경우, 유사 뉴스 카드 없이 아래 문장만 출력하세요.**

                저는 과거 뉴스 정보를 바탕으로 답변해드리는 AI 챗봇 뉴스토스입니다. 😄

                ---

                ### 2. 경제·산업·주식·정책 등 시사 이슈 질문
                - 반드시 아래 형식으로 뉴스 카드 HTML을 그대로 출력하세요.

                ---

                ## [뉴스 카드 출력 형식]

                ** 주의: 코드블록(```html ... ```) 사용 절대 금지!  
                HTML 태그를 그대로 출력하여 브라우저에서 실제 렌더링되도록 해야 함.**

                ### HTML 출력 예시:

                <div style="margin-bottom: 24px;">
                <h3 style="margin: 0 0 8px 0;">
                    <a href="https://n.news.naver.com/mnews/article/015/0005063326" target="_blank" style="text-decoration: none; color: #0070f3;">
                    하이브 상장 때 4000억 따로 챙긴 방시혁…당국, 제재 여부 검토
                    </a>
                </h3>

                <img src="https://imgnews.pstatic.net/image/015/2024/11/29/0005063326_001_20241129155613852.jpg?type=w200"
                    alt="뉴스 이미지"
                    style="width: 200px; border-radius: 8px; margin-bottom: 12px;">

                <ul style="margin: 0 0 12px 20px; padding: 0;">
                    <li><strong>유사도</strong>: 0.58</li>
                    <li><strong>날짜</strong>: 2024-11-29</li>
                    <li><strong>요약</strong>: 방시혁 하이브 의장은 2020년 하이브 상장 전 스틱인베스트먼트 등과 주주 간 계약을 맺고...</li>
                </ul>

                <p style="margin-top: 12px;">추가적으로 궁금한 점이 있으면 언제든 질문해 주세요 😉</p>
                </div>

                ---

                ## [유사 질문 추천]

                답변 마지막에는 반드시 아래 형식으로 **의문문 형태의 유사 질문** 2~3개를 출력하세요.  
                **마크다운 대신 아래 HTML 구조만 허용됩니다.**

                <h3>아래와 같은 질문도 함께 참고해보세요</h3>
                <ul>
                    <li>하이브 상장 당시 방시혁 의장의 계약 내용은 무엇이었나요?</li>
                    <li>과거 IPO 주관사 선정 과정에서 어떤 이슈들이 있었나요?</li>
                    <li>IPO 실패 시 지분 반환 조건이 적용된 사례가 있나요?</li>
                </ul>
            """

            user_prompt = f"""## [제공된 유사 뉴스 카드]
                    {context}

                    ## [사용자 질문]
                    {question}"""

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        else:
            system_prompt = """당신은 과거 뉴스 기반 AI 챗봇입니다.  
                    유사 뉴스가 없을 경우, 아래 3가지 예시 중 하나만 선택해 정중히 안내하세요.  
                    **절대 종합 정보, 배경 설명, 개인 의견을 추가하지 마세요.**

                    - "조금 더 구체적으로 질문해주시면, 최신 이슈와 다양한 데이터를 바탕으로 최선을 다해 답변해드릴게요!"
                    - "직접적인 연관 사례를 확인하려면, 더 구체적으로 질문해주세요. ☺️"
                    - "더 구체적으로 질문해주시면, 정확한 답변을 드릴 수 있습니다!"
                """
            user_prompt = f"""## [사용자 질문]{question}"""

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

    def make_stream_prompt(self, question, top_k=2):
        similar_news = self.search_similar_news(question, top_k=top_k)
        filtered_news = [row for row in similar_news if row.get("similarity", 0) >= 0.1]
        retrieved_infos = []
        for row in filtered_news:
            info = (
                f"{row['title']} ({row['url']})\n"
                f"<img src=\"{row['image']}\" alt=\"뉴스 이미지\">\n"
                f"{row['summary']}\n"
                f"{row['wdate'][:10]}\n"
                f"(유사도: {row.get('similarity', 0):.2f})"
            )
            retrieved_infos.append(info)

        context = "\n\n".join(retrieved_infos)
        return self.build_prompt(context, question, has_news=bool(filtered_news))

    def answer(self, question, top_k=2):
        messages = self.make_stream_prompt(question, top_k)

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=1024,
        )

        return response.choices[0].message.content


# 스케일러 로더: 각 파일을 딕셔너리로 읽어 들여서, 스케일링 시 변수명을 기준으로 접근할 수 있게 구성
def load_scalers_by_group(folder_path):
    scalers = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".joblib"):
            key = filename.replace(".joblib", "")
            full_path = os.path.join(folder_path, filename)
            obj = joblib.load(full_path)

            # 버전 정보 포함된 dict일 경우 대응
            if isinstance(obj, dict) and "scaler" in obj:
                scalers[key] = obj["scaler"]
            else:
                scalers[key] = obj

    return scalers


# 과거 유사 뉴스 검색을 위한 모델 로딩 함수
def get_similarity_model():
    model_dir = "models/"
    scaler_dir = os.path.join(model_dir, "scalers_grouped")
    ae_path = os.path.join(model_dir, "ae_encoder.onnx")
    regressor_path = os.path.join(model_dir, "similarity_ranker.onnx")

    # ONNX 모델 로딩
    ae_sess = ort.InferenceSession(ae_path)
    regressor_sess = ort.InferenceSession(regressor_path)

    # 스케일러 로딩
    scalers = load_scalers_by_group(scaler_dir)

    return scalers, ae_sess, regressor_sess


def get_recommend_model():
    model_base_path = Path("models")

    # ONNX 세션 생성
    model_recommend = ort.InferenceSession(
        str(model_base_path / "two_tower_model.onnx")
    )

    return model_recommend


def get_recommend_ranker_model():
    model_base_path = Path("models")

    # ONNX 세션 생성
    model_recommend_ranker = joblib.load(str(model_base_path / "lgbm_model2.pkl"))

    return model_recommend_ranker
