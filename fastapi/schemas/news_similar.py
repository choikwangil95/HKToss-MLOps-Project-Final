from pydantic import BaseModel
from datetime import date

class SimilarNews(BaseModel):
    news_id: int
    date: date
    title: str
    content: str
    url: str
    similarity: float