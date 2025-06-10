from sqlalchemy import Column, String, Date, Text, Integer, ARRAY, DateTime
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


class NewsModel_v2(Base):
    __tablename__ = "news_v2"

    news_id = Column(String, primary_key=True)  # VARCHAR PRIMARY KEY
    wdate = Column(DateTime)  # date
    title = Column(Text)  # title TEXT
    url = Column(Text)  # url TEXT
    article = Column(Text)  # content TEXT
    press = Column(Text)
    image = Column(Text)


class NewsModel_v2_Metadata(Base):
    __tablename__ = "news_v2_metadata"

    news_id = Column(String, primary_key=True)  # VARCHAR PRIMARY KEY
    summary = Column(Text)  # summary TEXT
    stock_list = Column(
        ARRAY(String)
    )  # stock_list VARCHAR, 쉼표로 구분된 종목명들 (예: "삼성전자, SK하이닉스")
    industry_list = Column(
        ARRAY(String)
    )  # industry_list VARCHAR, 쉼표로 구분된 업종명들 (예: "반도체, IT")


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
