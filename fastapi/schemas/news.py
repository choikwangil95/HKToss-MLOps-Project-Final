from pydantic import BaseModel
from datetime import datetime


class News(BaseModel):
    title: str
    content: str
    category: str
    published_at: datetime


class NewsOut(News):
    id: int
