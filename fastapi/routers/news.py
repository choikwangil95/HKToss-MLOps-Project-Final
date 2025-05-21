from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas.news import News, NewsOut, SimilarNews
from services.news import get_news_list, get_news_detail, find_news_similar
from core.db import get_db
from typing import List

router = APIRouter(prefix="/news", tags=["News"])


@router.get("/", response_model=list[NewsOut])
def list_news(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return get_news_list(db, skip, limit)


@router.get("/{news_id}", response_model=NewsOut)
def news_detail(news_id: str, db: Session = Depends(get_db)):
    return get_news_detail(db, news_id)


@router.get("/{news_id}/similar", response_model=List[SimilarNews])
def similar_news(
    news_id: str,
    top_n: int = 5,
    min_gap_days: int = 180,
    min_gap_between: int = 90,
    db: Session = Depends(get_db),
):
    results = find_news_similar(db, news_id, top_n, min_gap_days, min_gap_between)
    return results
