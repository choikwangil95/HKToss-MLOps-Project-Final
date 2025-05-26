from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from schemas.news import (
    News,
    NewsOut,
    SimilarNews,
    Report,
    NewsStock,
    PastReportsResponse,
)
from services.news import (
    get_news_list,
    get_news_detail,
    find_news_similar,
    get_similar_past_reports,
    find_stock_effected,
)
from core.db import get_db
from typing import List
import json
import numpy as np
from models.news import NewsModel


router = APIRouter(
    prefix="/stocks", tags=["Stocks"], responses={404: {"description": "Not found"}}
)


@router.get(
    "/{stock_id}/prediction/opinion",
    response_model=list[NewsOut],
    summary="[예정] 주식 종목 구매의견 예측 API",
    description="주식 종목 구매의견 예측 API.",
    include_in_schema=True,
)
def get_predicted_opinion():
    """
    주식 종목의 증권사 리포트 구매의견을 예측합니다.
    """
    return JSONResponse(
        status_code=501,
        content={"message": "주식 종목 구매의견 예측 API는 현재 준비 중입니다."},
    )


@router.get(
    "/{stock_id}/prediction/price",
    response_model=list[NewsOut],
    summary="[예정] 주식 종목 목표주가 예측 API",
    description="주식 종목 목표주가 예측 API.",
    include_in_schema=True,
)
def get_predicted_target_price():
    """
    주식 종목 목표주가를 예측합니다.
    """
    return JSONResponse(
        status_code=501,
        content={"message": "주식 종목 목표주가 예측 API는 현재 준비 중입니다."},
    )
