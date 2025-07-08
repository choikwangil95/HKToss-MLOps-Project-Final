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


class NewsTossChatbot:
    def __init__(self):
        # OpenAI 클라이언트 준비 (환경변수 OPENAI_API_KEY 필요)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_client(self):
        return self.client

    def search_similar_news(self, query_text, top_k=2):
        # Step 1: 첫 번째 API로 query_text 기반 가장 유사한 뉴스 1개 찾기
        first_url = "http://15.164.44.39:8000/news/similar"
        response = requests.post(first_url, json={"article": query_text, "top_k": 2})
        response.raise_for_status()
        top_news = response.json()["similar_news_list"]
        # news_id = top_news["news_id"]
        similar_news = top_news.copy()

        # Step 2: 해당 news_id를 두 번째 API에 넣어서 유사 뉴스 top_k개 가져오기
        # second_url = f"http://3.37.207.16:8000/news/v2/{news_id}/similar?top_n=2"
        # response = requests.get(second_url)
        # response.raise_for_status()
        # similar_news = response.json()

        return similar_news

    def build_prompt(self, context, question):
        system_prompt = """
            당신은 주식 투자에 도움을 주는 전문 AI 챗봇, '뉴스토스'입니다.  
            아래 3가지 질문 유형에 따라 답변 기준을 철저히 지키며 경제, 금융, 주식과 연관지어 답변하세요. 
            ---

            ## [질문 유형 및 답변 규칙]

            ### 1. 정체성 관련 질문
            - 예시: "너 누구야", "정체가 뭐야", "니 역할은 뭐야", "너 뭐하는 애야"
            - 정체가 뭔지, 어떤 역할을 하는지 묻는 경우, 유사 뉴스 카드 없이 아래와 같이 출력하세요.
            "저는 당신의 주식 투자에 도움을 주는 챗봇 '뉴스토스'입니다. 😄 <br>
            1. 캘린더를 확인하고, 앞으로 있을 일정과 관련된 과거 유사 뉴스를 물어보세요! <br>
            2. 경제, 금융 용어나 주식 투자 관련 궁금한 점을 물어보세요!"

            ### 2. 과거 유사 뉴스 질문(경제·산업·주식·정책 등)
            - 과거 뉴스를 알려달라는 질문에는 꼭 아래와 같이 답하세요.
            - 반드시 아래 형식으로 뉴스 카드 HTML을 그대로 출력하세요.
            - 가급적 2개의 뉴스 카드를 보여주세요.
            - 아래 출력 예시 속 뉴스 카드 내용은 참고만 하되, 동일하게 답변하지 마세요.
            - 사용자의 질문에서 기업명/종목명을 찾고, 그와 연관된 유사한 뉴스를 찾아 답변하세요.
                - 기업명/종목명이 없다면, 동일 산업군에서 유사 뉴스를 찾아 답변하세요.
            - context에 없는 뉴스, 날짜, 요약, 제목, 이미지는 절대 생성하지 마세요.

            [뉴스 카드 출력 형식]
            - ** 주의: 코드블록(```html ... ```) 사용 절대 금지!  
            - HTML 태그를 그대로 출력하여 브라우저에서 실제 렌더링되도록 해야 함.**
            - 뉴스 카드 제목 윗줄에 유사 뉴스 1️⃣, 유사 뉴스 2️⃣ 를 표시하세요.
            - HTML 출력 예시:

            
            <h3 style="margin: 0 0 8px 0; font-size: 20px !important;">
                <strong style="font-size: 20px !important;">유사 뉴스 1️⃣</strong><br>
                <a href="https://n.news.naver.com/mnews/article/015/0005063326" target="_blank" style="text-decoration: underline; color: #0070f3;">
                <strong>하이브 상장 때 4000억 따로 챙긴 방시혁…당국, 제재 여부 검토</strong>
                </a>
            </h3>

            <img src="https://imgnews.pstatic.net/image/015/2024/11/29/0005063326_001_20241129155613852.jpg?type=w200"
                alt="뉴스 이미지"
                style="width: 200px; border-radius: 8px; margin-bottom: 12px;">
    
                <p><strong>📊유사도</strong>: 0.58</p>
                <p><strong>🗓️날짜</strong>: 2024-11-29</p>
                <p><strong>📄요약</strong>: 방시혁 하이브 의장은 2020년 하이브 상장 전 스틱인베스트먼트 등과 주주 간 계약을 맺고...</p>
                <br>


            [유사 질문 추천]    
            - 유사 질문 추천은 뉴스 카드 답변 시에만 추가하고, 정체성 관련 질문이나 그 외 질문에는 노출하지 마세요.
            - 뉴스카드 답변 마지막에는 반드시 아래 형식으로 **의문문 형태의 유사 질문** 2~3개를 출력하고, 마무리 멘트를 넣어주세요.  
            - **마크다운 대신 아래 HTML 구조만 허용됩니다.**

            <br />
            <h3 style="margin-top: 10px; font-size: 20px !important;">아래와 같은 질문도 함께 참고해보세요!</h3>
            <br />
                <p>▸ 하이브 상장 당시 방시혁 의장의 계약 내용은 무엇이었나요?</p>
                <p>▸ 과거 IPO 주관사 선정 과정에서 어떤 이슈들이 있었나요?</p>
                <p>▸ IPO 실패 시 지분 반환 조건이 적용된 사례가 있나요?</p>
            </ul>
            <br />
            <p style="margin-top: 12px;">질문에 "회사 이름"과 "특정 사건/이슈"를 포함하면 답변 정확도가 올라가요!<br>
            더 궁금한 점이 있으면 언제든 질문해 주세요😉</p>


            ### 3. 그 외, 경제·금융 용어, 투자 전략, 주식 관련 일반 질문 등
            - 주식 투자에 도움이 되는 정보를 중심으로, GPT의 전문 지식을 활용해 자유롭게 답변하세요.
            - 사용자가 읽기 편하게끔 **문단 나누기**, **줄바꿈**, **강조 표시** 등을 적극적으로 이용하세요.
            - 답변을 HTML 형식으로 작성해주세요. 예를 들어 :
                - 중요한 부분은 <strong> 태그로 강조
                - 리스트는 <ul>, <li> 태그로 강조
                문단 사이에는 <p> 태그로 구분
            - 예시: 용어 해설, 투자 전략 설명, 금융 상품 비교, 시장 분석, 재무지표 해석, 투자 팁 등
            - 답변은 친절하고 명확하게, 초보자도 이해할 수 있도록 작성하세요.
            - **단, 시사 이슈/사건이 아니라면 뉴스카드 형식은 사용하지 마세요.**

            ---

            ## [금지사항]
            - 허위 정보, 미래 예측, 개인 의견, 투자 권유는 포함하지 마세요.
            - 항상 중립적이고 정보 중심적으로 답변하세요.

        """

        user_prompt = f"""## [제공된 유사 뉴스 카드]
                {context}

                ## [사용자 질문]
                {question}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def make_stream_prompt(self, question, top_k=2):
        similar_news = self.search_similar_news(question, top_k=top_k)
        # filtered_news = [row for row in similar_news if row.get("similarity", 0) >= 0.1]
        # filtered_news = similar_news.copy()
        retrieved_infos = []
        for row in similar_news:
            info = (
                f"{row['title']} ({row['url']})\n"
                f"<img src=\"{row['image']}\" alt=\"뉴스 이미지\">\n"
                f"{row['summary']}\n"
                f"{row['wdate'][:10]}\n"
                f"(유사도: {0.5 + row.get('similarity', 0):.2f})"
            )
            retrieved_infos.append(info)

        context = "\n\n".join(retrieved_infos)
        return self.build_prompt(context, question)

    def answer(self, question, top_k=2):
        messages = self.make_stream_prompt(question, top_k)

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
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


def get_recommend_ranker_model():
    model_base_path = Path("models")

    # ONNX 세션 생성
    model_recommend_ranker = joblib.load(str(model_base_path / "lgbm_model2.pkl"))

    return model_recommend_ranker
