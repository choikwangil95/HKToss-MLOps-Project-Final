from sqlalchemy import (
    INT,
    Column,
    String,
    Date,
    Text,
    Integer,
    ARRAY,
    DateTime,
    JSON,
    Float,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector


Base = declarative_base()


class UserProfileModel(Base):
    __tablename__ = "user_profile"

    user_id = Column(String, primary_key=True)
    user_pnl = Column("userpnl", Integer)  # ← DB에서는 "userPnl", Python에서는 user_pnl
    asset = Column(Integer)
    invest_score = Column("investscore", Integer)  # ← DB에서는 "investScore"
    member_stocks = Column("memberstocks", JSON)  # ← DB에서는 "memberStocks"


class UserLogModel(Base):
    __tablename__ = "news_v2_log"

    id = Column(INT, primary_key=True)
    user_id = Column(String)
    news_id = Column(String)
    wdate = Column(DateTime)  # date
