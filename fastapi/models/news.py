from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from core.db import Base


class NewsModel(Base):
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(50))
    published_at = Column(DateTime, default=datetime.utcnow)
