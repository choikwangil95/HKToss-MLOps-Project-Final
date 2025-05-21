from sqlalchemy import Column, String, Date, Text
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class NewsModel(Base):
    __tablename__ = "news"

    news_id = Column(String, primary_key=True)  # VARCHAR PRIMARY KEY
    date = Column(Date)  # date
    title = Column(Text)  # title TEXT
    url = Column(Text)  # url TEXT
    content = Column(Text)  # content TEXT
    embedding = Column(Vector(768))  # embedding VECTOR(768)
