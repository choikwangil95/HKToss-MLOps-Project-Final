from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas.news import Embedding
from services.news import get_news_embedding
from typing import List

router = APIRouter(prefix="/news", tags=["News"])


@router.get("/embedding", response_model=Embedding)
def get_news_embedding(content: str):
    return get_news_embedding(content)
