# news_report.py (또는 news.py에 포함 가능)
from fastapi import APIRouter, HTTPException
from services.news_report import get_similar_past_reports
from schemas.news_report import PastReportsResponse

router = APIRouter(prefix="/news", tags=["News"])

@router.get("/{news_id}/matched-reports", response_model=PastReportsResponse)
def matched_reports(news_id: int, topk: int = 5):
    try:
        results = get_similar_past_reports(news_id=news_id, topk=topk)
        return {"results": results}
    except IndexError:
        raise HTTPException(status_code=404, detail="존재하지 않는 뉴스 ID입니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")
