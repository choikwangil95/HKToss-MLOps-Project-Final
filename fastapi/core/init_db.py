from sqlalchemy.orm import Session
from core.db import Base, engine, get_db
from models.news import NewsModel
from datetime import datetime
import logging


def init_db():
    Base.metadata.create_all(bind=engine)  # 테이블 생성

    db: Session = next(get_db())

    # 이미 데이터가 있으면 건너뜀
    if db.query(NewsModel).first():
        logging.info("✅ News table already initialized.")
        return

    sample_news = NewsModel(
        title="AI 주도 투자 시대 개막",
        content="OpenAI와 같은 기업이 주도하는 AI 기술이 금융시장에 빠르게 확산 중입니다.",
        category="경제",
        published_at=datetime(2024, 12, 1, 12, 0, 0),
    )
    db.add(sample_news)
    db.commit()
    logging.info("✅ Sample news inserted.")
