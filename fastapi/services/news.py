from sqlalchemy.orm import Session
from sqlalchemy import desc
from models.news import NewsModel
from schemas.news import News
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import datetime
import json


def get_news_list(db: Session, skip: int = 0, limit: int = 20):
    news_list = (
        db.query(NewsModel)
        .order_by(desc(NewsModel.date))
        .offset(skip)
        .limit(limit)
        .all()
    )

    return news_list


def get_news_detail(db: Session, news_id: str):
    news = db.query(NewsModel).filter(NewsModel.news_id == news_id).first()

    if news is None:
        return None

    return news


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
