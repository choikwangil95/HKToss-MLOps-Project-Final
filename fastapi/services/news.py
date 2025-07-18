from sqlalchemy.orm import Session
from sqlalchemy import or_, desc, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import text
from models.news import (
    NewsModel,
    NewsModel_v2,
    NewsModel_v2_External,
    NewsModel_v2_Metadata,
    NewsModel_v2_Similarity,
    ReportModel,
)
from sqlalchemy import func
from datetime import datetime, date
from fastapi.responses import JSONResponse
from schemas.news import News, NewsOut_v2_External, SimilarNewsV2
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import datetime
import json
import ast
from fastapi import HTTPException
import requests
from datetime import datetime, timedelta
import random
import time

import pandas as pd
import ast
import numpy as np
import os
from pykrx import stock
import requests
from datetime import timedelta
from tqdm import tqdm
import threading

pykrx_lock = threading.Lock()

from dotenv import load_dotenv

load_dotenv()


def get_news_list(
    db: Session,
    skip: int = 0,
    limit: int = 20,
    title: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    query = db.query(NewsModel)

    if title:
        query = query.filter(NewsModel.title.ilike(f"%{title}%"))
    if start_date:
        query = query.filter(NewsModel.date >= start_date)
    if end_date:
        query = query.filter(NewsModel.date <= end_date)

    news_list = query.order_by(desc(NewsModel.news_id)).offset(skip).limit(limit).all()

    return news_list


def get_news_list_v2(
    db: Session,
    skip: int = 0,
    limit: int = 20,
    title: Optional[str] = None,
    stock_list: Optional[List[str]] = None,
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
):
    query = db.query(
        NewsModel_v2,
        NewsModel_v2_Metadata.stock_list,
        NewsModel_v2_Metadata.impact_score,
    ).join(NewsModel_v2_Metadata, NewsModel_v2.news_id == NewsModel_v2_Metadata.news_id)

    if title:
        query = query.filter(NewsModel_v2.title.ilike(f"%{title}%"))
    if start_datetime:
        query = query.filter(NewsModel_v2.wdate >= start_datetime)
    if end_datetime:
        query = query.filter(NewsModel_v2.wdate <= end_datetime)

    if stock_list:
        stock_conditions = [
            cast(NewsModel_v2_Metadata.stock_list, JSONB).contains(
                [{"stock_id": stock_id}]
            )
            for stock_id in stock_list
        ]
        if stock_conditions:
            subquery = (
                db.query(NewsModel_v2_Metadata.news_id)
                .filter(or_(*stock_conditions))
                .subquery(name="matched_news_ids")
            )
            query = query.filter(NewsModel_v2.news_id.in_(subquery))

    results = query.order_by(desc(NewsModel_v2.wdate)).offset(skip).limit(limit).all()

    response = []
    for news_obj, stock_list_value, impact_score in results:
        news_dict = {
            **news_obj.__dict__,
            "stock_list": stock_list_value,
            "impact_score": impact_score,
        }
        news_dict.pop("_sa_instance_state", None)
        response.append(news_dict)

    return response


def get_news_count(db: Session):
    # 전체 뉴스 개수
    total_count = db.query(func.count()).select_from(NewsModel_v2).scalar()

    # 오늘 뉴스 개수 (wdate가 datetime일 경우 날짜 비교)
    today = date.today()
    today_count = (
        db.query(func.count())
        .select_from(NewsModel_v2)
        .filter(func.date(NewsModel_v2.wdate) == today)
        .scalar()
    )

    return {
        "news_count_total": total_count,
        "news_count_today": today_count,
    }


def get_news_detail(db: Session, news_id: str):
    news = db.query(NewsModel).filter(NewsModel.news_id == news_id).first()

    if news is None:
        return None

    return news


def get_news_detail_v2(db: Session, news_id: str):
    news = db.query(NewsModel_v2).filter(NewsModel_v2.news_id == news_id).first()

    if news is None:
        return None

    return news


def get_news_detail_v2_metadata(db: Session, news_id: str):
    news = (
        db.query(NewsModel_v2_Metadata)
        .filter(NewsModel_v2_Metadata.news_id == news_id)
        .first()
    )

    if news is None:
        raise HTTPException(status_code=404, detail="News not found")

    return {
        "news_id": news.news_id,
        "summary": news.summary,
        "stock_list": news.stock_list,
        "stock_list_view": news.stock_list_view,
        "industry_list": news.industry_list,
        "impact_score": (
            f"{news.impact_score:.3f}" if news.impact_score is not None else None
        ),
    }


def extract_d_minus_1_info(news: dict) -> dict:
    # 날짜 처리
    news_date = pd.to_datetime(news["wdate"]).normalize()
    year, month = news_date.year, news_date.month

    # 전달 계산
    if month == 1:
        prev_year, prev_month = year - 1, 12
    else:
        prev_year, prev_month = year, month - 1

    # 거래일 수집 (해당 월 + 전달)
    trading_days = []
    for y, m in [(prev_year, prev_month), (year, month)]:
        try:
            days = stock.get_previous_business_days(year=y, month=m)
            trading_days.extend(days)
        except:
            continue
    trading_days = pd.to_datetime(sorted(set(trading_days)))

    # D-day (뉴스일 기준 가장 가까운 거래일)
    d_day_idx = trading_days.searchsorted(news_date, side="right") - 1
    if d_day_idx < 0:
        return {}

    d_day = trading_days[d_day_idx]
    d_minus_1_idx = d_day_idx - 1
    if d_minus_1_idx < 0:
        return {}

    d_minus_1 = trading_days[d_minus_1_idx]

    # Ticker 추출
    stock_list = news.get("stock_list", [])
    if not stock_list or not isinstance(stock_list, list):
        return {}

    ticker = str(stock_list[-1]["stock_id"]).zfill(6)

    # d-1 및 fallback 날짜 문자열 생성
    fallback_dates = [d_minus_1 - timedelta(days=i) for i in range(0, 10)]
    fallback_dates_str = [d.strftime("%Y%m%d") for d in fallback_dates]

    # OHLCV 수집
    try:
        ohlcv = stock.get_market_ohlcv_by_date(
            min(fallback_dates_str), max(fallback_dates_str), ticker
        ).reset_index()
        ohlcv.rename(columns={"날짜": "date"}, inplace=True)
        ohlcv["ticker"] = ticker
    except:
        ohlcv = pd.DataFrame()

    # 수급 데이터 수집
    try:
        trade = stock.get_market_trading_value_by_date(
            min(fallback_dates_str), max(fallback_dates_str), ticker
        ).reset_index()
        trade.rename(columns={"날짜": "date"}, inplace=True)
        trade["ticker"] = ticker
    except:
        trade = pd.DataFrame()

    # fallback: 가장 가까운 날짜의 값
    def get_latest(source_df, cols):
        for d in fallback_dates:
            row = source_df[(source_df["date"] == d) & (source_df["ticker"] == ticker)]
            if not row.empty:
                return row.iloc[0][cols].to_dict()
        return {col: None for col in cols}

    ohlcv_vals = get_latest(ohlcv, ["종가", "거래량"])
    trade_vals = get_latest(trade, ["개인", "기관합계", "외국인합계"])

    return {
        "news_id": news["news_id"],
        "d_minus_1_close": ohlcv_vals["종가"],
        "d_minus_1_volume": ohlcv_vals["거래량"],
        "d_minus_1_individual": trade_vals["개인"],
        "d_minus_1_institution": trade_vals["기관합계"],
        "d_minus_1_foreign": trade_vals["외국인합계"],
    }


def reconstruct_absolute_values_with_d1_base(external: dict, d1_base: dict) -> dict:
    import numpy as np

    result = external.copy()

    # D-1 기준값
    d1 = {
        "close": d1_base.get("d_minus_1_close", 0),
        "volume": d1_base.get("d_minus_1_volume", 0),
        "foreign": d1_base.get("d_minus_1_foreign", 0),
        "institution": d1_base.get("d_minus_1_institution", 0),
        "individual": d1_base.get("d_minus_1_individual", 0),
    }

    # D-1 값도 반영해줌
    for key, val in d1.items():
        result[f"d_minus_1_date_{key}"] = val

    # D-5 ~ D-2 역변환 (변화율 → 절대값)
    for i in range(5, 1, -1):
        prefix = f"d_minus_{i}"
        for k in d1.keys():
            pct = external.get(f"{prefix}_date_{k}", None)
            base = d1[k]
            if pct is not None and base != 0:
                result[f"{prefix}_date_{k}"] = round(base * (1 - pct), 2)

    # D+1 ~ D+5 역변환 (수익률 → 절대값)
    close_d1 = d1["close"]
    for i in range(1, 6):
        key = f"d_plus_{i}_date_close"
        pct = external.get(key, None)
        if pct is not None and close_d1 != 0:
            result[key] = round(close_d1 * (1 - pct), 2)

    return result


def retry(func, retries=3, delay=0.5):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(delay)


def get_news_detail_v2_external(db: Session, news_id: str):
    # 1. 뉴스 기본 정보 가져오기
    news = (
        db.query(
            NewsModel_v2.news_id,
            NewsModel_v2.wdate,
            NewsModel_v2_Metadata.stock_list,
        )
        .join(
            NewsModel_v2_Metadata,
            NewsModel_v2.news_id == NewsModel_v2_Metadata.news_id,
        )
        .filter(NewsModel_v2.news_id == news_id)
        .first()
    )

    if news is None:
        raise HTTPException(status_code=404, detail="News not found")

    news_dict = {
        "news_id": news[0],
        "wdate": news[1],
        "stock_list": news[2],
    }

    # 2. D-1 기준값 계산
    with pykrx_lock:
        d_minus_1_info = extract_d_minus_1_info(news_dict)
    print("[D-1 기준값]", d_minus_1_info)

    # 3. 외부 데이터 가져오기
    news_external = (
        db.query(NewsModel_v2_External)
        .filter(NewsModel_v2_External.news_id == news_id)
        .first()
    )

    if news_external is None:
        raise HTTPException(status_code=404, detail="External data not found")

    # 4. ORM → dict 변환
    news_external_dict = {
        col.name: getattr(news_external, col.name)
        for col in NewsModel_v2_External.__table__.columns
    }

    # 5 역변환
    data = reconstruct_absolute_values_with_d1_base(news_external_dict, d_minus_1_info)
    print("[복원된 데이터]", data)

    # 6 복원된 값으로 DTO 리턴
    return NewsOut_v2_External(**data)


def find_news_similar(
    db: Session, news_id: str, top_n=5, min_gap_days=180, min_gap_between=90
):
    # DB에서 타겟 뉴스 데이터 가져오기
    target_news = db.query(NewsModel).filter(NewsModel.news_id == news_id).first()
    if target_news is None:
        return []

    target_news_date = target_news.date
    target_news_embedding = target_news_embedding = np.array(
        target_news.embedding
    ).reshape(1, -1)

    # DB에서 유사 후보 뉴스 데이터 가져오기
    date_threshold = target_news_date - timedelta(days=min_gap_days)
    candidate_news = (
        db.query(NewsModel)
        .filter(NewsModel.news_id != news_id)
        .filter(NewsModel.date <= date_threshold)
        .all()
    )

    # 3. 임베딩 있는 후보만 정리
    candidates = []
    for news in candidate_news:
        if news.embedding is None:
            continue
        candidates.append(
            {
                "news_id": news.news_id,
                "date": news.date,
                "title": news.title,
                "content": news.content,
                "url": news.url,
                "embedding": news.embedding,
            }
        )

    if not candidates:
        return []

    # 4. cosine similarity 계산
    embedding_matrix = np.stack([c["embedding"] for c in candidates])
    similarities = cosine_similarity(target_news_embedding, embedding_matrix)[0]

    for i, sim in enumerate(similarities):
        candidates[i]["similarity"] = float(sim)

    # 5. 유사도 내림차순 정렬
    candidates = sorted(candidates, key=lambda x: x["similarity"], reverse=True)
    candidates = candidates[:top_n]

    # 6. 날짜 간격 조건에 맞게 Top-N 선택
    # selected = []
    # for row in candidates:
    #     if any(
    #         abs((row["date"] - sel["date"]).days) < min_gap_between for sel in selected
    #     ):
    #         continue
    #     selected.append(row)
    #     if len(selected) == top_n:
    #         break

    # 7. 결과 반환
    result = [
        {
            "news_id": row["news_id"],
            "date": row["date"].strftime("%Y-%m-%d"),
            "title": row["title"],
            "content": row["content"],
            "url": row["url"],
            "similarity": round(row["similarity"], 6),
        }
        for row in candidates
    ]

    return result


# 과거 뉴스-리포트 매칭 (유사도 top-k)
def get_similar_past_reports(
    db: Session,
    news_embedding: np.ndarray,
    news_date,
    topk: int = 5,
    min_similarity: float = 0.0,
    date_margin: int = 90,
):
    # 1. 임베딩이 있고 날짜가 있는 리포트만 조회
    report_qs = db.query(ReportModel).all()
    candidates = []
    candidate_embeddings = []
    candidate_dates = []

    for r in report_qs:
        if r.embedding is None or r.date is None:
            continue
        # 날짜 차이
        day_diff = (r.date - news_date).days
        if -date_margin <= day_diff <= date_margin:
            candidates.append(r)
            candidate_embeddings.append(np.array(r.embedding))
            candidate_dates.append(r.date)

    if not candidates:
        return []

    embeddings = np.stack(candidate_embeddings)
    sims = cosine_similarity(news_embedding.reshape(1, -1), embeddings)[0]

    # 유사도 내림차순 topk
    top_indices = sims.argsort()[::-1][:topk]

    results = []
    for idx in top_indices:
        r = candidates[idx]
        target_price = r.target_price if r.target_price is not None else ""
        emb = r.embedding if r.embedding is not None else []
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        results.append(
            {
                "report_id": r.report_id,
                "stock_name": r.stock_name,
                "title": r.title,
                "sec_firm": r.sec_firm,
                "date": r.date.strftime("%Y-%m-%d"),
                "view_count": r.view_count,
                "url": r.url,
                "target_price": target_price,
                "opinion": r.opinion,
                "report_content": r.report_content,
                "similarity": float(sims[idx]),
            }
        )
    return results


def find_stock_effected(db: Session, news_id: str):
    news = db.query(NewsModel).filter(NewsModel.news_id == news_id).first()

    if news is None:
        return None

    # str 타입의 stocks를 리스트로 변환
    stocks = "".join(news.stocks)
    stocks = ast.literal_eval(stocks) if isinstance(stocks, str) else stocks

    return [{"news_id": news.news_id, "stocks": stocks}] if news.stocks else []


def get_top_impact_news(
    db: Session,
    start_datetime: datetime,
    end_datetime: datetime,
    limit: int = 10,
    stock_list: Optional[List[str]] = None,
) -> list[dict]:
    """특정 기간 내 상위 impact_score 뉴스 조회"""
    if start_datetime >= end_datetime:
        raise ValueError("시작일은 종료일보다 앞서야 합니다.")

    # 기본 쿼리
    query = (
        db.query(
            NewsModel_v2.news_id,
            NewsModel_v2.wdate,
            NewsModel_v2.title,
            NewsModel_v2.image,
            NewsModel_v2.press,
            NewsModel_v2.url,
            NewsModel_v2_Metadata.summary,
            NewsModel_v2_Metadata.impact_score,
            NewsModel_v2_Metadata.stock_list,
        )
        .join(
            NewsModel_v2_Metadata, NewsModel_v2.news_id == NewsModel_v2_Metadata.news_id
        )
        .filter(
            NewsModel_v2.wdate >= start_datetime,
            NewsModel_v2.wdate < end_datetime,
            NewsModel_v2_Metadata.impact_score.isnot(None),
        )
    )

    # 종목 필터링 (JSONB contains 방식)
    if stock_list:
        stock_conditions = [
            cast(NewsModel_v2_Metadata.stock_list, JSONB).contains(
                [{"stock_id": stock_id}]
            )
            for stock_id in stock_list
        ]
        if stock_conditions:
            subquery = (
                db.query(NewsModel_v2_Metadata.news_id)
                .filter(or_(*stock_conditions))
                .subquery(name="matched_news_ids")
            )
            query = query.filter(NewsModel_v2.news_id.in_(subquery))

    # 정렬 및 결과 추출
    results = (
        query.order_by(NewsModel_v2_Metadata.impact_score.desc()).limit(limit).all()
    )

    return [
        {
            "news_id": row.news_id,
            "wdate": row.wdate,
            "title": row.title,
            "image": row.image,
            "press": row.press,
            "summary": row.summary,
            "impact_score": row.impact_score,
            "url": row.url,
            "stock_list": row.stock_list,  # ✅ 포함
        }
        for row in results
    ]


def find_news_similar_v3(
    db: Session, news_id: str, top_n: int, min_gap_days: int, min_gap_between: int
):
    # DB에서 정보 조회
    results = (
        db.query(NewsModel_v2_Similarity)
        .filter(NewsModel_v2_Similarity.news_id == news_id)
        .order_by(desc(NewsModel_v2_Similarity.similarity))
        .all()
    )

    def convert(row):
        # ✅ 메타데이터에서 stock_list 추가 조회
        stock_list = (
            db.query(NewsModel_v2_Metadata.stock_list)
            .filter(NewsModel_v2_Metadata.news_id == row.sim_news_id)
            .scalar()
        )

        return {
            "news_id": row.sim_news_id,
            "wdate": row.wdate.isoformat() if row.wdate else None,
            "title": row.title,
            "press": row.press,
            "url": row.url,
            "image": row.image,
            "summary": row.summary,
            "similarity": row.similarity,
            "stock_list": stock_list,  # ✅ 포함
        }

    return [convert(r) for r in results[:top_n]]


def find_news_similar_v2(
    db: Session, news_id: str, top_n: int, min_gap_days: int, min_gap_between: int
) -> List[SimilarNewsV2]:
    # 기준 뉴스 조회
    ref_news_meta = (
        db.query(NewsModel_v2_Metadata)
        .filter(NewsModel_v2_Metadata.news_id == news_id)
        .first()
    )
    ref_news_raw = (
        db.query(NewsModel_v2).filter(NewsModel_v2.news_id == news_id).first()
    )

    if not ref_news_raw:
        return []

    ref_wdate = ref_news_raw.wdate

    # 기준 뉴스 텍스트 추출
    # text = ref_news_meta.summary if ref_news_meta else ref_news_raw.article[:300]
    text = ref_news_raw.title + ref_news_raw.article[:300]
    if not text.strip():
        return []

    # 유사 뉴스 API 호출
    try:
        response = requests.post(
            "http://15.165.211.100:8000/news/similar",
            json={"article": text, "top_k": 10},
        )
        response.raise_for_status()
        similar_news_list = response.json()["similar_news_list"]
    except Exception as e:
        print(f"유사 뉴스 API 요청 실패: {e}")
        return []

    # 필터링 조건 적용
    min_date = ref_wdate - timedelta(days=min_gap_days)

    def is_far_enough(new_date: datetime, selected_dates: List[datetime]) -> bool:
        return all(abs((new_date - d).days) >= min_gap_between for d in selected_dates)

    filtered_output = []
    selected_dates = []

    similar_news_list = sorted(
        similar_news_list, key=lambda x: x["similarity"], reverse=True
    )

    for item in similar_news_list:
        item_date = datetime.fromisoformat(item["wdate"])
        if (
            item["similarity"]
            < 0.9
            # and item_date <= min_date
            # and is_far_enough(item_date, selected_dates)
        ):
            filtered_output.append(item)
            selected_dates.append(item_date)
        # if len(filtered_output) >= top_n:
        # break

    similar_news_ids = [item["news_id"] for item in filtered_output]
    filtered_ids = [nid for nid in similar_news_ids if nid != news_id]

    # 유사 뉴스 Rerank API 호출
    try:
        response = requests.post(
            "http://15.165.211.100:8000/news/similarity",
            json={
                "news_id": ref_news_raw.news_id,
                "news_topk_ids": filtered_ids,
            },
        )
        response.raise_for_status()
        similar_news_reranked_list = response.json()["results"]
    except Exception as e:
        print(f"유사 뉴스 Rerank API 요청 실패: {e}")
        print(f"텍스트 유사도만 조회하도록 합니다.: {e}")

        similar_news_reranked_list = filtered_output
        # return []

    # filtered_output = []
    # selected_dates = []

    # for item in similar_news_reranked_list:
    #     item_date = datetime.fromisoformat(item["wdate"])
    #     if (
    #         item["similarity"] < 0.9
    #         and item_date <= min_date
    #         and is_far_enough(item_date, selected_dates)
    #     ):
    #         filtered_output.append(item)
    #         selected_dates.append(item_date)
    #     if len(filtered_output) >= top_n:
    #         break

    # similar_news_reranked_list = filtered_output

    # 유사 뉴스 요약 맵
    summary_map = {
        item["news_id"]: {
            "summary": item["summary"],
            "similarity": item.get("score") or item.get("similarity"),
        }
        for item in similar_news_reranked_list
    }

    similar_ids = list(summary_map.keys())

    # DB에서 메타 정보 조회
    results = (
        db.query(
            NewsModel_v2.news_id,
            NewsModel_v2.wdate,
            NewsModel_v2.title,
            NewsModel_v2.image,
            NewsModel_v2.press,
            NewsModel_v2.url,
        )
        .filter(NewsModel_v2.news_id.in_(similar_ids))
        .all()
    )

    # SimilarNewsV2 객체 생성
    output = []
    for row in results:
        meta = summary_map.get(row.news_id)
        if meta:
            output.append(
                SimilarNewsV2(
                    news_id=row.news_id,
                    wdate=row.wdate.isoformat(),
                    title=row.title,
                    press=row.press,
                    url=row.url,
                    image=row.image,
                    summary=meta["summary"],
                    similarity=round(meta["similarity"], 3),
                )
            )

    # 유사도 높은 순 정렬
    output.sort(key=lambda x: x.similarity, reverse=True)

    return output[:top_n]


def collect_member_news_data(
    member_id: str, start_date: str, end_date: str
) -> (list, pd.DataFrame):  # type: ignore
    """
    특정 멤버의 뉴스 로그 데이터 수집
    :param member_id: 멤버 ID (예: "anonymous", "user123")
    :param start_date: 조회 시작 날짜 (YYYY-MM-DD)
    :param end_date: 조회 종료 날짜 (YYYY-MM-DD)
    :return:
        - unique_news_ids: 중복 제거된 newsId 목록
        - click_log_df: 원본 로그 데이터 DataFrame (news_id 컬럼 포함)
    """
    API_BASE_URL = "http://3.39.99.26:8080"
    NEWS_LOGS_ENDPOINT = "/api/newsLogs"
    url = API_BASE_URL + NEWS_LOGS_ENDPOINT
    params = {"startDate": start_date, "endDate": end_date, "memberId": member_id}

    try:
        response = requests.get(url, params=params, timeout=1)
        response.raise_for_status()
        api_response = response.json()
    except Exception as e:
        print(f"API 호출 실패: {str(e)}")
        return [], pd.DataFrame()

    if not api_response.get("Success", False):
        return [], pd.DataFrame()

    # 원본 데이터 추출
    log_data = api_response.get("data", [])

    # 1. 뉴스 ID 추출 및 중복 제거
    news_ids = [log.get("newsId") for log in log_data if "newsId" in log]
    unique_news_ids = list(set(news_ids))

    # 2. 원본 로그 데이터를 DataFrame으로 변환
    click_log_df = pd.DataFrame(log_data)

    # 3. 컬럼명 변경: newsId -> news_id
    if "newsId" in click_log_df.columns:
        click_log_df = click_log_df.rename(columns={"newsId": "news_id"})

    if member_id == None:
        return [], pd.DataFrame()

    print(
        f"멤버 '{member_id}': {len(log_data)}개 로그, {len(unique_news_ids)}개 고유 뉴스"
    )
    return unique_news_ids, click_log_df


async def get_news_recommended(user_id, db):
    start_all = time.perf_counter()

    use_other_user = False
    user_click_count = 0
    other_user_data = None

    # 1. 클릭 로그 조회
    t0 = time.perf_counter()

    end_datetime = datetime.now().replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    start_datetime = (end_datetime - timedelta(days=14)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_date = start_datetime.strftime("%Y-%m-%d")
    end_date = end_datetime.strftime("%Y-%m-%d")

    unique_news_ids, click_log_df = collect_member_news_data(
        user_id, start_date, end_date
    )

    user_click_count = len(unique_news_ids)

    print(f"[TIME] 사용자 클릭 로그 수집: {time.perf_counter() - t0:.3f}s")

    # 2. 유사 사용자 대체 로직
    if len(unique_news_ids) < 5:
        t1 = time.perf_counter()

        use_other_user = True

        # 현재 사용자
        try:
            response = requests.get(
                f"http://3.39.99.26:8080/api/v1/userinfo/{user_id}", timeout=1
            )
            response.raise_for_status()
            user_data = response.json()["data"]
        except Exception as e:
            print(f"사용자 {user_id} 정보 조회 실패: {str(e)}")
            user_data = {}

        try:
            user_data_all = requests.get("http://3.37.207.16:8000/users").json()
        except Exception as e:
            print(f"사용자 목록 정보 조회 실패: {str(e)}")
            user_data_all = []

        user_invest_score = user_data.get("investScore", 1)
        if user_invest_score == 0:
            user_invest_score = user_invest_score + 1

        matched_user = next(
            (u for u in user_data_all if u["invest_score"] == user_invest_score), None
        )

        if matched_user:
            other_user_data = matched_user.copy()
            user_id = matched_user["user_id"]

            try:
                user_data_logs = requests.get(
                    f"http://3.37.207.16:8000/users/{user_id}/logs"
                ).json()

                unique_news_ids = list({data["news_id"] for data in user_data_logs})
                unique_news_ids = random.sample(
                    unique_news_ids, min(20, len(unique_news_ids))
                )
            except Exception as e:
                print(f"유사 사용자 뉴스 로그 조회 실패: {str(e)}")
        else:
            print("유사 user_id를 찾을 수 없습니다.")

        print(f"[TIME] 유사 사용자 처리: {time.perf_counter() - t1:.3f}s")

    # 3. 주요 뉴스 가져오기
    t2 = time.perf_counter()
    top_news = get_top_impact_news(
        db=db,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        limit=30,
        stock_list=None,
    )
    print(f"[TIME] 주요 뉴스 조회: {time.perf_counter() - t2:.3f}s")

    # 4. 후보 필터링 모델 호출
    t3 = time.perf_counter()
    clicked_news_ids = unique_news_ids.copy()
    candidate_news_ids = [news["news_id"] for news in top_news]

    try:
        response = requests.post(
            "http://15.164.44.39:8000/news/recommend",
            json={
                "news_clicked_ids": clicked_news_ids[:10],
                "news_candidate_ids": candidate_news_ids,
            },
        )
        response.raise_for_status()
        news_recomended_candidates_ids = response.json()
    except Exception as e:
        print(f"추천 뉴스 후보군 API 호출 실패: {str(e)}")
        news_recomended_candidates_ids = []

    print(f"[TIME] 추천 후보 필터링 모델 호출: {time.perf_counter() - t3:.3f}s")

    print(f"[DEBUG] 리랭킹 대상 뉴스 수: {len(news_recomended_candidates_ids)}")
    # 5. 리랭킹 모델 호출
    t4 = time.perf_counter()
    try:
        response = requests.post(
            "http://15.164.44.39:8000/news/recommend/rerank",
            json={"user_id": user_id, "news_ids": news_recomended_candidates_ids},
        )
        response.raise_for_status()
        news_recomended_list = response.json()
    except Exception as e:
        print(f"추천 뉴스 리랭킹 API 호출 실패: {str(e)}")
        news_recomended_list = []

    print(f"[TIME] 리랭킹 모델 호출: {time.perf_counter() - t4:.3f}s")

    print(f"[TIME] 전체 추천 소요 시간: {time.perf_counter() - start_all:.3f}s")

    return {
        "user_click_count": user_click_count,
        "use_other_user": use_other_user,
        "other_user_data": other_user_data,
        "news_data": news_recomended_list,
    }
