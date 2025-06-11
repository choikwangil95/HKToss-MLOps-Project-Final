from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime, date
from typing import List, Union
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
        orm_mode = True


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
        orm_mode = True


class NewsOut_v2(BaseModel):
    news_id: str
    wdate: Optional[datetime]
    title: str
    article: str
    url: Optional[str]
    press: str
    image: str

    class Config:
        orm_mode = True


class NewsOut_v2_Metadata(BaseModel):
    news_id: str
    summary: str
    stock_list: Optional[List[str]] = []  # None이면 빈 리스트
    industry_list: Optional[List[str]] = []  # None이면 빈 리스트
    impact_score: Optional[float]

    class Config:
        orm_mode = True

    @field_validator("stock_list", "industry_list", mode="before")
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
        orm_mode = True


class PastReportsResponse(BaseModel):
    results: List[Report]
