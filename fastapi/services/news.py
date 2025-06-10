from sqlalchemy.orm import Session
from sqlalchemy import desc
from models.news import NewsModel, NewsModel_v2, NewsModel_v2_Metadata, ReportModel
from schemas.news import News
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from typing import Optional
import datetime
import json
import ast
from fastapi import HTTPException
from datetime import datetime


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
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
):
    query = db.query(NewsModel_v2)

    if title:
        query = query.filter(NewsModel_v2.title.ilike(f"%{title}%"))

    if start_datetime:
        query = query.filter(NewsModel_v2.wdate >= start_datetime)
    if end_datetime:
        query = query.filter(NewsModel_v2.wdate <= end_datetime)

    news_list = query.order_by(desc(NewsModel_v2.wdate)).offset(skip).limit(limit).all()

    return news_list


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
        "industry_list": news.industry_list,
    }


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
