import numpy as np
import re
from konlpy.tag import Okt
import numpy as np
from sqlalchemy.orm import Session
from collections import Counter

from schemas.model import SimilarNewsItem
import asyncio
from fastapi.responses import StreamingResponse
import json
import redis
from concurrent.futures import ThreadPoolExecutor
import requests
from db.postgresql import get_db
from models.custom import (
    NewsModel_v2,
    NewsModel_v2_Metadata,
    NewsModel_v2_External,
    NewsModel_v2_Topic,
)
import shap
import json

import ast
import pandas as pd
import time
from datetime import datetime


async def get_news_embeddings(article_list, request):
    """
    뉴스 본문 리스트를 임베딩하는 함수입니다.
    ONNX 모델이 배치 입력을 지원할 경우, 한 번에 추론합니다.
    """
    tokenizer = request.app.state.tokenizer_embedding
    session = request.app.state.session_embedding

    # 1. 토큰화
    encoded = [tokenizer.encode(x) for x in article_list]
    input_ids = [e.ids for e in encoded]
    attention_mask = [[1] * len(ids) for ids in input_ids]

    # 2. 패딩 (최대 길이 기준)
    max_len = max(len(ids) for ids in input_ids)
    input_ids_padded = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
    attention_mask_padded = [
        mask + [0] * (max_len - len(mask)) for mask in attention_mask
    ]

    # 3. numpy 배열로 변환
    input_ids_np = np.array(input_ids_padded, dtype=np.int64)
    attention_mask_np = np.array(attention_mask_padded, dtype=np.int64)

    # 4. ONNX 추론
    outputs = session.run(
        ["sentence_embedding"],
        {"input_ids": input_ids_np, "attention_mask": attention_mask_np},
    )[
        0
    ]  # shape: (batch_size, hidden_dim)

    # 5. 반환 (List[List[float]])
    return outputs.tolist()


def safe_parse_list(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return val if isinstance(val, list) else []


async def get_news_similar_list(payload, request):
    """
    유사 뉴스 top_k
    """
    article = payload.article
    top_k = payload.top_k

    vectordb = request.app.state.vectordb

    # 검색
    results = vectordb.similarity_search_with_score(article, k=100)

    news_similar_list = []
    seen_titles = set()

    for doc, score in results:
        similarity = round(1 - float(score), 2)

        if similarity > 0.9:
            continue  # 유사도가 너무 높으면 제외 (0.9 이상 필터링)

        title = doc.metadata.get("title")
        if title in seen_titles:
            continue  # 이미 추가한 title이면 스킵
        seen_titles.add(title)

        news_id = doc.metadata.get("news_id")
        wdate = doc.metadata.get("wdate")
        summary = doc.page_content
        url = doc.metadata.get("url")
        image = doc.metadata.get("image")
        stock_list = safe_parse_list(doc.metadata.get("stock_list"))
        industry_list = safe_parse_list(doc.metadata.get("industry_list"))

        if not news_id or not wdate or not summary:
            continue

        news_similar_list.append(
            SimilarNewsItem(
                news_id=news_id,
                wdate=wdate,
                title=title,
                summary=summary,
                url=url,
                image=image,
                stock_list=stock_list,
                industry_list=industry_list,
                similarity=similarity,
            )
        )

    return news_similar_list[:top_k]


# 전역 Redis 연결
redis_conn = redis.Redis(
    host="3.39.99.26",
    port=6379,
    password="q1w2e3r4!@#",
    decode_responses=True,
)

# 전역 ThreadPoolExecutor
redis_executor = ThreadPoolExecutor(max_workers=10)


# Redis 전송 함수
def send_to_redis(data: dict):
    try:
        redis_conn.publish("chat-response", json.dumps(data, ensure_ascii=False))
    except Exception as e:
        print(f"[Redis Error] {type(e).__name__}: {e}")


# 🔁 전체 처리 로직 (백그라운드에서 실행됨)
def process_chat_and_publish(chatbot, payload):
    try:
        start = time.time()
        print(f"[{datetime.now()}] 🔵 Background 작업 시작")

        t1 = time.time()
        messages = chatbot.make_stream_prompt(payload.question, top_k=2)
        print(
            f"[{datetime.now()}] ✅ make_stream_prompt 완료 ({time.time() - t1:.2f}s)"
        )

        t2 = time.time()
        stream = chatbot.get_client().chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=1024,
            stream=True,
        )
        print(f"[{datetime.now()}] ✅ GPT 스트리밍 준비 완료 ({time.time() - t2:.2f}s)")

        idx = 0
        for chunk in stream:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                data = {
                    "client_id": payload.client_id,
                    "content": content,
                    "is_last": False,
                    "index": idx,
                }
                send_to_redis(data)
                idx += 1

        # 마지막 메시지
        data = {
            "client_id": payload.client_id,
            "content": "",
            "is_last": True,
            "index": idx,
        }
        send_to_redis(data)
        print(
            f"[{datetime.now()}] ✅ Redis 전송 완료 (총 {idx+1}개, {time.time() - start:.2f}s 소요)"
        )

    except Exception as e:
        print(f"[❌ Background Error] {type(e).__name__}: {e}")


# ✅ FastAPI 엔드포인트 함수
async def get_stream_response(request, payload):
    print(f"[{datetime.now()}] 🟢 요청 수신 → 백그라운드 실행 시작")
    chatbot = request.app.state.chatbot
    loop = asyncio.get_event_loop()

    # 전체 처리 백그라운드 실행
    loop.run_in_executor(None, process_chat_and_publish, chatbot, payload)

    print(f"[{datetime.now()}] 🟢 응답 즉시 반환")
    return {"message": "처리 중입니다. Redis 채널(chat-response)로 전송됩니다."}


async def get_embedding_batch(article_list, request):
    """문장 리스트를 배치로 임베딩 처리"""
    batch_size = 20

    all_embeddings = []
    for i in range(0, len(article_list), batch_size):
        batch = article_list[i : i + batch_size]
        try:
            batch_news_embeddings = await get_news_embeddings(batch, request)
            batch_embeddings = batch_news_embeddings if batch_news_embeddings else []
            all_embeddings.extend(batch_embeddings)
            print(f"배치 {i}-{i+len(batch)}: {len(batch_embeddings)}개 임베딩 성공")

        except Exception as e:
            print(f"배치 {i}-{i+len(batch)} 실패: {str(e)}")
            all_embeddings.extend([None] * len(batch))

    return all_embeddings


async def get_news_metadata_df(news_ids: list) -> pd.DataFrame:
    """
    뉴스 ID 목록을 받아 각 뉴스의 메타데이터를 조회하여 DataFrame으로 반환
    :param news_ids: 뉴스 ID 문자열 리스트 (예: ['20250513_0092', '20250618_28735495'])
    :return: 뉴스 메타데이터 DataFrame
    """
    API_BASE_URL = "http://3.37.207.16:8000"
    METADATA_ENDPOINT = "/news/v2/{news_id}/metadata"

    records = []

    # news_ids가 유효한 리스트인지 확인
    if not isinstance(news_ids, list) or not all(isinstance(i, str) for i in news_ids):
        print("잘못된 news_ids 형식: 반드시 문자열 리스트여야 합니다")
        return pd.DataFrame()

    for news_id in news_ids:
        # URL 정상 생성 확인
        url = API_BASE_URL + METADATA_ENDPOINT.format(news_id=news_id)
        try:
            response = requests.get(url)
            response.raise_for_status()  # 4xx/5xx 오류 시 예외 발생
            meta = response.json()
            records.append(meta)
            print(f"{news_id} 메타데이터 조회 성공")
        except requests.exceptions.HTTPError as e:
            print(f"{news_id} 처리 중 HTTP 오류 ({e.response.status_code}): {e}")
        except Exception as e:
            print(f"{news_id} 처리 중 오류: {e}")

    return pd.DataFrame(records)


async def get_news_recommended(payload, request):
    """
    뉴스 후보군 추천
    """
    news_clicked_ids = payload.news_clicked_ids
    news_candidate_ids = payload.news_candidate_ids

    # 임베딩
    news_clicked_metadata = await get_news_metadata_df(news_clicked_ids)
    news_candidate_medatata = await get_news_metadata_df(news_candidate_ids)

    news_clicked_summaries = news_clicked_metadata["summary"].tolist()
    news_candidate_summaries = news_candidate_medatata["summary"].tolist()

    news_clicked_embeddings = await get_embedding_batch(news_clicked_summaries, request)
    news_candidate_embeddings = await get_embedding_batch(
        news_candidate_summaries, request
    )

    # 임베딩 데이터 추출 및 전처리
    clicked_embeddings_np = np.array(news_clicked_embeddings, dtype=np.float32)
    candi_embeddings_np = np.array(news_candidate_embeddings, dtype=np.float32)

    # 입력 데이터 형식 조정
    num_clicked = len(news_clicked_embeddings)
    num_candidates = len(news_candidate_embeddings)

    clicked_input = clicked_embeddings_np[:num_clicked].reshape(1, num_clicked, 768)
    candidate_input = candi_embeddings_np[:num_candidates].reshape(
        1, num_candidates, 768
    )

    # 추론 Step 1
    top_k = 5  # 클릭 뉴스 별 후보 뉴스 유사 확률 top_k

    # 모델 추론 실행
    inputs = {"clicked": clicked_input, "candidates": candidate_input}
    model_recommend = request.app.state.model_recommend
    outputs = model_recommend.run(None, inputs)[0]

    scores = outputs  # [1, 5]
    top_indices = np.argsort(-scores, axis=2)[:, :, :top_k]
    flattened = top_indices.squeeze(0).flatten()

    # 유니크 후보군 및 등장 빈도 계산
    unique_candidates = np.unique(flattened)
    counts = Counter(flattened)

    # [0, 2, 6, 4, ...]
    candidate_indices = unique_candidates.tolist()

    # 실제 뉴스 ID 매핑
    top_k_news_ids = [news_candidate_ids[i] for i in candidate_indices]

    # 최종 반환
    # ['20250513_0089', '20250513_0148', '20250513_0085', ... ]
    return top_k_news_ids


async def get_news_recommended_ranked(payload, request, db):
    """
    뉴스 후보군 추천
    """
    model_recommend_ranker = request.app.state.model_recommend_ranker
    user_id = payload.user_id

    # 기본값
    user_data = {"userPnl": 0, "asset": 0, "investScore": 1}

    is_user_exist = user_id not in [None, "", "None"]
    if is_user_exist:
        # 1차 API 요청 (3.39.99.26)
        try:
            url1 = f"http://3.39.99.26:8080/api/v1/userinfo/{user_id}"
            response = requests.get(url1, timeout=1)
            response.raise_for_status()
            data = response.json()["data"]

            if isinstance(data, dict) and data:
                user_data = data
            else:
                raise ValueError(f"✅ 사용자 {user_id} 정보 조회 실패 (1차): {url1}")
        except Exception as e:
            print(f"❌ 1차 사용자 API 실패: {str(e)}")

            # 2차 API 요청 (3.37.207.16)
            try:
                url2 = f"http://3.37.207.16:8000/users/{user_id}"
                response = requests.get(url2, timeout=1)
                response.raise_for_status()
                user_data = response.json()

                print(f"✅ 사용자 {user_id} 정보 조회 성공 (2차): {url2}")
            except Exception as e:
                print(f"❌ 2차 사용자 API 실패: {str(e)}")
                print(f"⚠️ 기본 사용자 데이터로 대체: {user_data}")

        user_data = {
            "userPnl": user_data.get("userPnl") or user_data.get("user_pnl", 0),
            "asset": user_data.get("asset", 0),
            "investScore": user_data.get("investScore")
            or user_data.get("invest_score", 1),
            "memberStocks": user_data.get("memberStocks")
            or user_data.get("member_stocks", []),
        }

    # 뉴스 정보 가져오기
    news_ids = payload.news_ids

    # 1. 각각 테이블에서 데이터 조회
    news_list = db.query(NewsModel_v2).filter(NewsModel_v2.news_id.in_(news_ids)).all()
    metadata_list = (
        db.query(NewsModel_v2_Metadata)
        .filter(NewsModel_v2_Metadata.news_id.in_(news_ids))
        .all()
    )
    external_list = (
        db.query(NewsModel_v2_External)
        .filter(NewsModel_v2_External.news_id.in_(news_ids))
        .all()
    )
    topic_list = (
        db.query(NewsModel_v2_Topic)
        .filter(NewsModel_v2_Topic.news_id.in_(news_ids))
        .all()
    )

    # 2. 각 테이블 결과를 news_id 기준 dict로 매핑
    news_map = {row.news_id: row.__dict__ for row in news_list}
    metadata_map = {row.news_id: row.__dict__ for row in metadata_list}
    external_map = {row.news_id: row.__dict__ for row in external_list}
    topic_map = {row.news_id: row.__dict__ for row in topic_list}

    # 3. SQLAlchemy 내부 키 제거 (_sa_instance_state)
    def clean_sqlalchemy_dict(d):
        return {k: v for k, v in d.items() if k != "_sa_instance_state"}

    # 4. 병합
    merged_results = []

    for news_id in news_ids:
        row = {}

        if news_id in news_map:
            row.update(clean_sqlalchemy_dict(news_map[news_id]))
        if news_id in metadata_map:
            row.update(clean_sqlalchemy_dict(metadata_map[news_id]))
        if news_id in external_map:
            row.update(clean_sqlalchemy_dict(external_map[news_id]))
        if news_id in topic_map:
            row.update(clean_sqlalchemy_dict(topic_map[news_id]))

        # [2] 사용자 데이터 병합
        row.update(user_data)  # user_data는 각 row에 동일하게 적용된다고 가정

        # [3] main_topic 계산
        topic_columns = [f"topic_{i}" for i in range(1, 10)]
        topic_values = [row.get(col) for col in topic_columns]

        if any(pd.notna(val) for val in topic_values):
            try:
                max_topic_index = int(np.argmax(topic_values)) + 1  # 1부터 시작
                row["main_topic"] = max_topic_index
            except Exception:
                row["main_topic"] = None
        else:
            row["main_topic"] = None

        # [4] is_same_stock 계산
        try:
            news_stocks = {
                stock["stock_id"] for stock in row.get("stock_list_view", [])
            }
            user_stocks = {stock["stockCode"] for stock in row.get("memberStocks", [])}
            row["is_same_stock"] = 1 if news_stocks & user_stocks else 0
        except Exception:
            row["is_same_stock"] = 0

        # [5] 병합 결과 저장
        merged_results.append(row)

    columns_to_keep = [
        "userPnl",
        "asset",
        "investScore",
        "topic_1",
        "topic_2",
        "topic_3",
        "topic_4",
        "topic_5",
        "topic_6",
        "topic_7",
        "topic_8",
        "topic_9",
        "main_topic",
        "is_same_stock",
    ]

    # 병합된 결과에서 필요한 컬럼만 남기기
    filtered_results = [
        {k: row.get(k, None) for k in columns_to_keep} for row in merged_results
    ]

    df_input = pd.DataFrame(filtered_results)

    try:
        # 클릭 확률 예측 (클래스 1에 대한 확률)
        proba = model_recommend_ranker.predict_proba(df_input)[:, 1]

        # 예측 결과를 원본 데이터프레임에 병합
        df_pred = pd.DataFrame(merged_results)
        df_pred["click_score"] = proba
        df_pred = df_pred[
            [
                "news_id",
                "wdate",
                "title",
                "summary",
                "image",
                "press",
                "url",
                "click_score",
                "stock_list",
            ]
        ]

        # 7. 확률 기준 정렬
        df_pred = df_pred.sort_values("click_score", ascending=False).reset_index(
            drop=True
        )

        # 5. SHAP value 계산
        explainer = shap.TreeExplainer(model_recommend_ranker)
        shap_values = explainer.shap_values(df_input)  # shape: [n_samples, n_features]

        # 6. 상위 3개 중요 피처명 추출
        feature_kor_map = {
            "userPnl": "투자 수익률 고려",
            "asset": "자산 보유량 고려",
            "investScore": "투자 성향 고려",
            "topic_1": "글로벌 시장 동향",
            "topic_2": "증권사 주가 전망",
            "topic_3": "기업 실적 개선",
            "topic_4": "증권사 사업 분석",
            "topic_5": "대기업 사업 전략",
            "topic_6": "기업 기술 개발",
            "topic_7": "반도체 및 AI",
            "topic_8": "금융 서비스",
            "topic_9": "주주 및 경영 이슈",
            "is_same_stock": "관심 유사 종목",
        }

        top_features_list = []
        feature_names = df_input.columns.tolist()

        for i, shap_row in enumerate(shap_values):
            abs_values = [abs(val) for val in shap_row]
            top_indices = sorted(range(len(abs_values)), key=lambda i: -abs_values[i])[
                :3
            ]
            top_feature_names = [feature_names[i] for i in top_indices]

            main_topic_value = df_input.iloc[i]["main_topic"]
            converted = []

            for f in top_feature_names:
                if f == "main_topic" and 1 <= int(main_topic_value) <= 9:
                    topic_key = f"topic_{int(main_topic_value)}"
                    converted.append(feature_kor_map.get(topic_key, topic_key))
                else:
                    converted.append(feature_kor_map.get(f, f))

            unique_converted = list(
                dict.fromkeys(converted)
            )  # ✅ 순서 유지 + 중복 제거
            top_features_list.append(unique_converted)

        # 7. SHAP 상위 피처 컬럼 추가
        df_pred["recommend_reasons"] = top_features_list

        news_recommend_ranked_list = df_pred[:10].to_dict(orient="records")

        return news_recommend_ranked_list

    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        return []
