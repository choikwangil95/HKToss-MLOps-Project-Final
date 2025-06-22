from fastapi import Query
from pydantic import BaseModel, field_validator, Field
from typing import Optional
from datetime import datetime, date
from typing import List, Union, Dict
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
    articles: List[str]


class EmbeddingOut(BaseModel):
    embeddings: List[List[float]] = Field(
        default_factory=list, description="(n, 768) 형식의 임베딩 벡터 리스트"
    )


class SimilarNewsIn(BaseModel):
    article: str
    top_k: int = Query(5, description="가장 유사한 뉴스 개수", ge=1, le=100)


class SimilarNewsItem(BaseModel):
    news_id: str
    wdate: str
    title: str
    summary: str
    url: Optional[str] = None
    image: Optional[str] = None
    stock_list: Optional[List[Dict[str, str]]] = []
    industry_list: Optional[List[Dict[str, str]]] = []
    similarity: float


class SimilarNewsOut(BaseModel):
    similar_news_list: List[SimilarNewsItem]


class LdaTopicsIn(BaseModel):
    article: str


class LdaTopicsOut(BaseModel):
    lda_topics: Dict[str, float]  # 예: {"topic1": 0.15, "topic2": 0.03, ...}


class ChatIn(BaseModel):
    client_id: str
    question: str


class ChatOut(BaseModel):
    client_id: str
    answer: str


# 요청: 단일 뉴스 ID 입력
class SimilarityRequest(BaseModel):
    news_id: str  # 사용자가 입력하는 값
    news_topk_ids: Optional[List[str]] = Field(
        default=None,
        description="기준 뉴스와 유사한 뉴스 ID 목록",
    )


# 응답: 유사 뉴스 5개에 대한 유사도 예측 점수 및 랭킹
class SimilarityResult(BaseModel):
    news_id: str
    summary: str
    score: float
    rank: int


# 응답 모델: 유사도 예측 결과를 포함하는 리스트
class SimilarityResponse(BaseModel):
    results: List[SimilarityResult]


class RecommendIn(BaseModel):
    news_clicked_ids: List
    news_candidate_ids: List


class RecommendOut(BaseModel):
    news_recommended: List


class RecommendRankedIn(BaseModel):
    user_id: str
    news_ids: List


class RecommendRankedOut(BaseModel):
    news_id: str
    wdate: datetime  # 날짜+시간
    title: str
    summary: str
    image: str | None  # 이미지 URL (nullable)
    press: str | None  # 언론사 (nullable)
    url: Optional[str]
    click_score: float
    recommend_reasons: List
