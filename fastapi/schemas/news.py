from fastapi import Query
from pydantic import BaseModel, field_validator, Field
from typing import Optional
from datetime import datetime, date
from typing import List, Union, Dict
from pydantic import validator
import ast


class News(BaseModel):
    news_id: str
    date: Optional[date]
    title: str
    url: Optional[str]
    content: str
    stocks: Optional[str]

    class Config:
        from_attributes = True


# stocks만 빼고
class NewsOut(BaseModel):
    news_id: str
    date: Optional[date]
    title: str
    url: Optional[str]
    content: str


class NewsStock(BaseModel):
    news_id: str
    stocks: List[str]


class SimilarNews(BaseModel):
    news_id: str
    date: date
    title: str
    content: str
    url: str
    similarity: float


#############################


def parse_comma_separated_stock_list(
    stock_list: Optional[str] = Query(
        None, description="콤마(,)로 구분된 종목코드 리스트 (예: 005930,000660)"
    )
) -> Optional[List[str]]:
    if stock_list is None or stock_list.strip() == "":
        return None
    return [s.strip() for s in stock_list.split(",") if s.strip()]


class News_v2(BaseModel):
    news_id: str
    wdate: Optional[datetime]
    title: str
    article: str
    url: Optional[str]
    press: str
    image: str
    stocks: Optional[str]

    class Config:
        from_attributes = True


class NewsOut_v2(BaseModel):
    news_id: str
    wdate: Optional[datetime]
    title: str
    article: str
    url: Optional[str]
    press: str
    image: str

    class Config:
        from_attributes = True


class NewsOut_v2_Metadata(BaseModel):
    news_id: str
    summary: str
    stock_list: Optional[List[Dict[str, str]]] = []
    stock_list_view: Optional[List[Dict[str, str]]] = []
    industry_list: Optional[List[Dict[str, str]]] = []
    impact_score: Optional[float]

    class Config:
        from_attributes = True

    @field_validator("stock_list", "stock_list_view", "industry_list", mode="before")
    @classmethod
    def parse_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except Exception:
                return [v]
        return v


class NewsOut_v2_External(BaseModel):
    news_id: str

    d_minus_5_date_close: Optional[float]
    d_minus_5_date_volume: Optional[float]
    d_minus_5_date_foreign: Optional[float]
    d_minus_5_date_institution: Optional[float]
    d_minus_5_date_individual: Optional[float]

    d_minus_4_date_close: Optional[float]
    d_minus_4_date_volume: Optional[float]
    d_minus_4_date_foreign: Optional[float]
    d_minus_4_date_institution: Optional[float]
    d_minus_4_date_individual: Optional[float]

    d_minus_3_date_close: Optional[float]
    d_minus_3_date_volume: Optional[float]
    d_minus_3_date_foreign: Optional[float]
    d_minus_3_date_institution: Optional[float]
    d_minus_3_date_individual: Optional[float]

    d_minus_2_date_close: Optional[float]
    d_minus_2_date_volume: Optional[float]
    d_minus_2_date_foreign: Optional[float]
    d_minus_2_date_institution: Optional[float]
    d_minus_2_date_individual: Optional[float]

    d_minus_1_date_close: Optional[float]
    d_minus_1_date_volume: Optional[float]
    d_minus_1_date_foreign: Optional[float]
    d_minus_1_date_institution: Optional[float]
    d_minus_1_date_individual: Optional[float]

    d_plus_1_date_close: Optional[float]
    d_plus_2_date_close: Optional[float]
    d_plus_3_date_close: Optional[float]
    d_plus_4_date_close: Optional[float]
    d_plus_5_date_close: Optional[float]

    fx: Optional[float]
    bond10y: Optional[float]
    base_rate: Optional[float]

    class Config:
        from_attributes = True  # ORM 지원


#############################


class Report(BaseModel):
    report_id: Optional[int]
    stock_name: str
    title: str
    sec_firm: str
    date: str
    view_count: int
    url: str
    target_price: Optional[str] = ""
    opinion: str
    report_content: str
    embedding: Optional[List[float]] = []
    similarity: Optional[float] = None

    class Config:
        from_attributes = True


class PastReportsResponse(BaseModel):
    results: List[Report]


class TopNewsResponse(BaseModel):
    news_id: str
    wdate: datetime  # 날짜+시간
    title: str
    image: str | None  # 이미지 URL (nullable)
    press: str | None  # 언론사 (nullable)
    summary: str
    impact_score: float
    url: Optional[str]

    class Config:
        from_attributes = True


class RecommendNewsResponse(BaseModel):
    news_id: str
    wdate: datetime  # 날짜+시간
    title: str
    summary: str
    image: str | None  # 이미지 URL (nullable)
    press: str | None  # 언론사 (nullable)
    url: Optional[str]
    click_score: float
    recommend_reasons: List

    class Config:
        from_attributes = True


class SimilarNewsV2(BaseModel):
    news_id: str
    wdate: Optional[datetime]
    title: str
    press: str
    url: str
    image: str
    summary: str
    similarity: float

    class Config:
        from_attributes = True


class RecommendedNewsV2(BaseModel):
    news_id: str
    wdate: str  # ISO format으로 주기 위해 str 처리
    title: str
    article: str
    press: str
    url: str
    image: str

    class Config:
        orm_mode = True
