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
            system_prompt = """당신은 과거 뉴스 정보를 바탕으로만 답변하는 전문 AI 챗봇, '뉴스토스'입니다.
                    - "너 누구야?"라고 묻는다면: "저는 과거 뉴스 정보를 바탕으로 답변해드리는 챗봇 뉴스토스입니다.😄" 라고 대답하세요.
                    - 아래 지침을 반드시 따르며, **절대 종합 정보나 추가 해석은 포함하지 마세요.**

                    ## [답변 작성 지침]

                    1. 반드시 아래와 같은 **뉴스 카드 형태**로 유사 뉴스 정보를 요약해 제시하세요 (순서 고정):
                        - 이미지 URL의 `type=w800`을 `type=w200`으로 반드시 수정하세요.
                        - 카드 내용은 다음 마크다운 예시처럼 작성하세요:
                        ```markdown
                        <img src="https://imgnews.pstatic.net/image/008/2024/11/28/0005120417_001_20241128085813446.jpg?type=w200" alt="뉴스 이미지">
                        ▶ **제목**: <a href="https://n.news.naver.com/mnews/article/008/0005120417" target="_blank">SK하이닉스, 신규 주주환원책으로 재무구조 개선 기대</a><br>  
                        ▶ **유사도**: 0.56   
                        ▶ **날짜**: 2024-11-28  
                        ▶ **요약**: NH투자증권이 신규 주주환원 정책을 공시한 SK하이닉스에 대해...

                        추가적으로 궁금한 점이 있으면 언제든 질문해 주세요😉
                        ```

                    2. 뉴스 카드 외의 해석, 예측, 종합적 의견은 **절대 포함하지 마세요.**

                    3. 답변 마지막에 다음 문장을 반드시 추가하고, **사용자 질문과 유사한 질문을 마크다운 불릿 리스트로 2~3개 생성**하세요:

                    **아래와 같은 질문도 함께 참고해 보실 수 있어요.**
                    - 사용자의 질문 주제를 유지한 채, 챗봇이 잘 대답할 수 있는 질문을 작성해야 합니다.
                    - 의문문 구조를 따라야 하며, 주제를 벗어나지 마세요.
                    """

            user_prompt = f"""## [제공된 유사 뉴스 카드]
                    {context}

                    ## [사용자 질문]
                    {question}"""

            return [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
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
                {"role": "user", "content": user_prompt}
            ]


    def make_stream_prompt(self, question, top_k=2):
        similar_news = self.search_similar_news(question, top_k=top_k)
        filtered_news = [row for row in similar_news if row.get('similarity', 0) >= 0.1]
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
    regressor_path = os.path.join(model_dir, "regressor_model.onnx")

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
