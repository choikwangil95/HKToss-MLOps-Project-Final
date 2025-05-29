from sqlalchemy import Column, String, Date, Text, Integer, ARRAY
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
    stocks = Column(
        ARRAY(String)
    )  # stocks VARCHAR, 쉼표로 구분된 종목명들 (예: "삼성전자, SK하이닉스")


class ReportModel(Base):
    __tablename__ = "reports"

    report_id = Column(
        Integer, primary_key=True, autoincrement=True
    )  # PK, 없으면 추가!
    stock_name = Column(String)
    title = Column(Text)
    sec_firm = Column(String)
    date = Column(Date)
    view_count = Column(Integer)
    url = Column(Text)
    target_price = Column(String)  # 쉼표포함 가격 - String
    opinion = Column(String)
    report_content = Column(Text)
    embedding = Column(Vector(768))
