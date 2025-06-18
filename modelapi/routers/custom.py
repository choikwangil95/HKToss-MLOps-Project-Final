from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Path,
    Request,
    Body,
    Response,
)
from sqlalchemy.orm import Session
from db.postgresql import get_db
from schemas.model import (
    ChatIn,
    ChatOut,
    LdaTopicsIn,
    LdaTopicsOut,
    RecommendIn,
    RecommendOut,
    SimilarNewsIn,
    SimilarNewsOut,
)
from services.model import (
    get_lda_topic,
    get_news_recommended,
    get_news_similar_list,
    get_stream_response,
)
from services.custom import get_news_impact_score_service

from schemas.custom import SimpleImpactResponse

import numpy as np

from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import openai
import os
import json

router = APIRouter(
    prefix="/news",
    tags=["Custom Model"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/similar",
    response_model=SimilarNewsOut,
    summary="유사 뉴스 top-k",
    description="유사 뉴스 top-k",
)
async def get_news_embedding_router(request: Request, payload: SimilarNewsIn):
    """
    유사 뉴스 top-k
    """
    article = payload.article
    if not article:
        raise HTTPException(
            status_code=400,
            detail="기사 본문이 비어있습니다. 올바른 본문을 입력해주세요.",
        )

    similar_news_list = get_news_similar_list(payload, request)
    if similar_news_list is None:
        raise HTTPException(
            status_code=500,
            detail="유사 뉴스 조회 중 오류가 발생했습니다. 다시 시도해주세요.",
        )

    return {"similar_news_list": similar_news_list}


@router.post(
    "/topics",
    response_model=LdaTopicsOut,
    summary="뉴스 요약문 LDA topic",
    description="뉴스 요약문 LDA topic",
)
async def get_news_summary_router(request: Request, payload: LdaTopicsIn):
    """
    뉴스 요약문 LDA topic
    """
    article = payload.article
    if not article:
        raise HTTPException(
            status_code=400,
            detail="기사 요약문이 비어있습니다. 올바른 본문을 입력해주세요.",
        )

    lda_topics = get_lda_topic(article, request)

    return {"lda_topics": lda_topics}


@router.post(
    "/chat/stream",
    # response_model=ChatOut,
    summary="뉴스 GPT 챗봇",
    description="뉴스 GPT 챗봇",
)
async def chat_stream_endpoint(request: Request, payload: ChatIn):
    return await get_stream_response(request, payload)


@router.post(
    "/recommend",
    response_model=RecommendOut,
    summary="뉴스 추천 후보군",
    description="뉴스 추천 후보군",
)
async def get_news_recommend(request: Request, payload: RecommendIn):
    return JSONResponse(
        status_code=200,
        content={"message": "🚧 현재 추천 API는 개발 중입니다. 곧 제공될 예정이에요!"},
    )

    # return await get_news_recommended(payload, request)


@router.get(
    "/{news_id}/impact_score",
    response_model=SimpleImpactResponse,
    summary="뉴스 ID로 뉴스 임팩트 스코어 계산",
    description="뉴스 ID만 입력하면 해당 뉴스의 임팩트 스코어를 반환합니다.",
)
async def get_news_impact_score(
    request: Request,
    response: Response,  # ✅ 추가
    news_id: str = Path(..., description="뉴스 고유 ID", min_length=1),
    db: Session = Depends(get_db),
):
    """
    특정 뉴스의 임팩트 스코어를 조회합니다.
    """
    d_plus, d_minus, impact_score, z_scores = await get_news_impact_score_service(
        news_id, db, request
    )  # request 전달

    # ✅ z_scores를 헤더에 JSON 형식으로 추가
    z_score_mean = float(np.mean(z_scores))
    response.headers["X-model-score"] = str(z_score_mean)  # Prometheus용 헤더 추가

    return SimpleImpactResponse(
        d_plus=d_plus, d_minus=d_minus, impact_score=impact_score
    )
