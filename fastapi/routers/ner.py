from fastapi import APIRouter, Depends, Path, HTTPException
from sqlalchemy.orm import Session
from schemas import StockMatchResponse
from services import match_stocks_by_news_id
from db import get_db  

router = APIRouter()

@router.get(
    '/{news_id}/stocks',
    response_model=StockMatchResponse,
    summary='뉴스 ID 기반 종목 추출',
    description='뉴스 ID를 입력하면 DB에서 해당 뉴스 본문을 불러와 종목명을 추출합니다.'
)
def match_news_to_stocks(
    news_id: str = Path(..., description='뉴스 ID'),
    db: Session = Depends(get_db)
):
    try:
        return match_stocks_by_news_id(news_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
