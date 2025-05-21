from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date


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
