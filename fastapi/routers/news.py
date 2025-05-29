from fastapi import APIRouter, Depends, HTTPException, Query, Path
from datetime import datetime
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from schemas.news import (
    News,
    News_v2,
    NewsOut,
    NewsOut_v2,
    SimilarNews,
    Report,
    NewsStock,
    PastReportsResponse,
)
from services.news import (
    get_news_list,
    get_news_list_v2,
    get_news_detail,
    get_news_detail_v2,
    find_news_similar,
    get_similar_past_reports,
    find_stock_effected,
)
from core.db import get_db
from typing import List
import json
from typing import Optional
import numpy as np
from models.news import NewsModel, NewsModel_v2


router = APIRouter(
    prefix="/news", tags=["News"], responses={404: {"description": "Not found"}}
)


@router.get(
    "/",
    response_model=list[NewsOut],
    summary="[완료] 뉴스 목록 조회 (test)",
    description="최신 뉴스 기사를 페이지 단위로 조회합니다.",
)
def list_news(
    skip: int = Query(0, description="건너뛸 뉴스 개수 (페이지네이션용)", ge=0),
    limit: int = Query(20, description="가져올 뉴스 개수", le=100),
    title: Optional[str] = Query(
        None, description="뉴스 제목 필터링 (부분 일치)", max_length=20
    ),
    start_date: Optional[str] = Query(None, description="시작 날짜 (예: 2025-05-01)"),
    end_date: Optional[str] = Query(None, description="종료 날짜 (예: 2025-05-31)"),
    db: Session = Depends(get_db),
):
    """
    뉴스 목록을 조회합니다.

    - `skip`: 건너뛸 뉴스 수 (기본값: 0)
    - `limit`: 최대 뉴스 수 (기본값: 20)
    - `title`: 뉴스 제목으로 필터링 (부분 일치)
    - `start_date`: 조회 시작 날짜 (형식: YYYY-MM-DD)
    - `end_date`: 조회 종료 날짜 (형식: YYYY-MM-DD)
    """
    return get_news_list(
        db,
        skip=skip,
        limit=limit,
        title=title,
        start_date=start_date,
        end_date=end_date,
    )


@router.get(
    "/v2",
    response_model=list[NewsOut_v2],
    summary="[완료] 뉴스 목록 조회 (production)",
    description="최신 뉴스 기사를 페이지 단위로 조회합니다.",
)
def list_news_v2(
    skip: int = Query(0, description="건너뛸 뉴스 개수 (페이지네이션용)", ge=0),
    limit: int = Query(20, description="가져올 뉴스 개수", le=100),
    title: Optional[str] = Query(
        None, description="뉴스 제목 필터링 (부분 일치)", max_length=50
    ),
    start_datetime: Optional[datetime] = Query(
        None, description="시작 일시 (예: 2025-05-01T00:00:00)"
    ),
    end_datetime: Optional[datetime] = Query(
        None, description="종료 일시 (예: 2025-05-31T23:59:59)"
    ),
    db: Session = Depends(get_db),
):
    """
    뉴스 목록을 조회합니다.
    - `title`: 뉴스 제목 부분 일치
    - `start_date`: 조회 시작 시각 (ISO 8601, 예: 2025-05-01T00:00:00)
    - `end_date`: 조회 종료 시각 (ISO 8601, 예: 2025-05-31T23:59:59)
    """
    return get_news_list_v2(
        db=db,
        skip=skip,
        limit=limit,
        title=title,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )


@router.get(
    "/highlights",
    response_model=list[NewsOut],
    summary="[예정] 주요 뉴스 목록 조회",
    description="주요 뉴스 기사를 조회합니다.",
    include_in_schema=True,
)
def get_highlighted_news():
    """
    주요 뉴스 목록을 조회합니다.
    """
    return JSONResponse(
        status_code=501,
        content={"message": "주요 뉴스 목록 조회 API는 현재 준비 중입니다."},
    )


@router.get(
    "/{news_id}",
    response_model=NewsOut,
    summary="[완료] 뉴스 상세 조회 (test)",
    description="뉴스 ID를 기반으로 해당 뉴스 기사의 상세 정보를 조회합니다.",
)
def news_detail(
    news_id: str = Path(..., description="뉴스 고유 ID"),
    db: Session = Depends(get_db),
):
    """
    특정 뉴스의 상세 정보를 조회합니다.
    """
    return get_news_detail(db, news_id)


@router.get(
    "/v2/{news_id}",
    response_model=NewsOut_v2,
    summary="[완료] 뉴스 상세 조회 (production)",
    description="뉴스 ID를 기반으로 해당 뉴스 기사의 상세 정보를 조회합니다.",
)
def news_detail(
    news_id: str = Path(..., description="뉴스 고유 ID"),
    db: Session = Depends(get_db),
):
    """
    특정 뉴스의 상세 정보를 조회합니다.
    """
    return get_news_detail_v2(db, news_id)


@router.get(
    "/{news_id}/related/news",
    response_model=List[SimilarNews],
    summary="[완료] 뉴스 관련 과거 유사 뉴스 조회",
    description="입력한 뉴스와 유사한 과거 뉴스를 조건에 따라 필터링하여 조회합니다.",
)
def similar_news(
    news_id: str = Path(..., description="기준이 되는 뉴스 ID"),
    top_n: int = Query(5, description="가장 유사한 뉴스 개수", ge=1, le=50),
    min_gap_days: int = Query(
        180, description="기준 뉴스와 유사 뉴스 간 최소 시간 간격 (일 단위)"
    ),
    min_gap_between: int = Query(
        90, description="유사 뉴스 간 최소 시간 간격 (일 단위)"
    ),
    db: Session = Depends(get_db),
):
    """
    기준 뉴스와 유사한 과거 뉴스 목록을 반환합니다.

    조건:
    - 기준 뉴스와 최소 `{min_gap_days}`일 이상 떨어져 있어야 함
    - 유사 뉴스끼리는 `{min_gap_between}`일 이상 떨어져 있어야 함
    """
    return find_news_similar(db, news_id, top_n, min_gap_days, min_gap_between)


@router.get(
    "/{news_id}/related/reports",
    response_model=PastReportsResponse,
    summary="[완료] 뉴스 관련 증권사 리포트 조회",
    description="특정 뉴스와 유사한 증권사 리포트를 조회합니다.",
)
def matched_reports(
    news_id: str = Path(..., description="뉴스 고유 ID"),
    topk: int = Query(5, description="리포트 Top-K 개수"),
    db: Session = Depends(get_db),
):
    news = db.query(NewsModel).filter(NewsModel.news_id == news_id).first()
    if not news or news.embedding is None or news.date is None:
        raise HTTPException(status_code=404, detail="뉴스 혹은 임베딩/날짜 없음")

    embedding = news.embedding
    # 임베딩 타입(str or list/array) 체크
    if isinstance(embedding, str):
        news_embedding = np.array(json.loads(embedding))
    else:
        news_embedding = np.array(embedding)
    news_embedding = news_embedding.reshape(1, -1)

    # news.date 전달
    results = get_similar_past_reports(
        db, news_embedding, news.date, topk=topk, date_margin=90
    )
    return {"results": results}


@router.get(
    "/{news_id}/related/stocks",
    response_model=List[NewsStock],
    summary="[완료] 뉴스 관련 주식 종목 조회",
    description="특정 뉴스와 관련된 주식 종목을 조회합니다.",
)
def matched_stock(
    news_id: str = Path(..., description="뉴스 고유 ID"), db: Session = Depends(get_db)
):
    """
    특정 뉴스와 관련된 주식 종목을 조회합니다.
    """

    return find_stock_effected(db, news_id)
