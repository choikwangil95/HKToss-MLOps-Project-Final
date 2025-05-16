from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from schemas.news import News, NewsOut
from services.news import get_news_list, get_news_detail
from core.db import get_db

router = APIRouter(prefix="/news", tags=["News"])


@router.get("/", response_model=list[NewsOut])
def list_news(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return get_news_list(db, skip, limit)


@router.get("/{news_id}", response_model=NewsOut)
def news_detail(news_id: int, db: Session = Depends(get_db)):
    return get_news_detail(db, news_id)
