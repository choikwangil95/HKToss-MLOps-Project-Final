from sqlalchemy.orm import Session
from models.news import NewsModel
from schemas.news import News
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
from services.model_cache import get_model, get_news_df
import pandas as pd

def get_news_list(db: Session, skip: int = 0, limit: int = 20):
    return db.query(NewsModel).offset(skip).limit(limit).all()


def get_news_detail(db: Session, news_id: int):
    return db.query(NewsModel).filter(NewsModel.id == news_id).first()


def find_news_similar(news_id: str, top_n=5, min_gap_days=180, min_gap_between=90):

    model = get_model()
    news_df = get_news_df()

    target_row = news_df[news_df['news_id'] == news_id]
    if target_row.empty:
        return []

    target_content = target_row.iloc[0]['content']

    target_date = pd.to_datetime(target_row.iloc[0]['date'])
    target_embedding = model.encode([target_content])

    news_df['date'] = pd.to_datetime(news_df['date']) 
    candidates = news_df[news_df['date'] <= target_date - timedelta(days=min_gap_days)].copy()

    candidates['similarity'] = cosine_similarity(target_embedding, list(candidates['embedding']))[0]
    candidates = candidates.sort_values(by='similarity', ascending=False)

    selected = []
    for _, row in candidates.iterrows():
        if any(abs((row['date'] - sel['date']).days) < min_gap_between for sel in selected):
            continue
        selected.append(row)
        if len(selected) == top_n:
            break

    result = [
        {
            'news_id': row['news_id'],
            'date': row['date'].strftime('%Y-%m-%d'),
            'title': row['title'],
            'content': row['content'],
            'url': row['url'],
            'similarity': round(row['similarity'], 6)
        }
        for row in selected
    ]
    return result
