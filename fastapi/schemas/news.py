from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date
from typing import List , Union

class News(BaseModel):
    news_id: str
    date: Optional[date]
    title: str
    url: Optional[str]
    content: str

    class Config:
        orm_mode = True


class NewsOut(News):
    pass  # 현재는 추가 필드가 없으므로 그대로 둡니다.


class SimilarNews(BaseModel):
    news_id: str
    date: date
    title: str
    content: str
    url: str
    similarity: float

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

    class Config:
        orm_mode = True

class PastReportsResponse(BaseModel):
    results: List[Report]
