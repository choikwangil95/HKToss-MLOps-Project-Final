from sqlalchemy.orm import Session
from models.news import NewsModel
from schemas.news import News


def get_news_list(db: Session, skip: int = 0, limit: int = 20):
    return db.query(NewsModel).offset(skip).limit(limit).all()


def get_news_detail(db: Session, news_id: int):
    return db.query(NewsModel).filter(NewsModel.id == news_id).first()
