from fastapi import Query
from pydantic import BaseModel, field_validator, Field
from typing import Optional
from datetime import datetime, date
from typing import List, Union
from pydantic import validator
import ast


class SummaryIn(BaseModel):
    article: str


class SummaryOut(BaseModel):
    summary: str


class StockIn(BaseModel):
    article: str


class StockOut(BaseModel):
    stock_list: List[str] = Field(
        default_factory=list,
        description="종목명 리스트",
    )


class EmbeddingIn(BaseModel):
    article: str


class EmbeddingOut(BaseModel):
    embedding: List[List[float]] = Field(
        default_factory=list, description="(1, 768) 형식의 임베딩 벡터"
    )


class SimilarNewsIn(BaseModel):
    article: str
    top_k: int = Query(5, description="가장 유사한 뉴스 개수", ge=1, le=100)


class SimilarNewsItem(BaseModel):
    news_id: str
    summary: str
    wdate: str
    score: float


class SimilarNewsOut(BaseModel):
    similar_news_list: List[SimilarNewsItem]
