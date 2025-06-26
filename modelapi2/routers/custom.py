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
    RecommendRankedIn,
    RecommendRankedOut,
    SimilarNewsIn,
    SimilarNewsOut,
    SimilarityRequest,
    SimilarityResponse,
    SimilarityResult,
)
from services.model import (
    get_news_recommended,
    get_news_recommended_ranked,
    get_news_similar_list,
    get_stream_response,
)

from schemas.custom import SimpleImpactResponse

import numpy as np

from sqlalchemy.orm import Session
from db.postgresql import get_db


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
    "/chat/stream",
    # response_model=ChatOut,
    summary="뉴스 GPT 챗봇",
    description="뉴스 GPT 챗봇",
)
async def chat_stream_endpoint(request: Request, payload: ChatIn):
    return await get_stream_response(request, payload)


@router.post(
    "/recommend",
    response_model=list[str],
    summary="뉴스 추천 후보군",
    description="뉴스 추천 후보군",
)
async def get_news_recommend(request: Request, payload: RecommendIn):
    return await get_news_recommended(payload, request)


@router.post(
    "/recommend/rerank",
    response_model=list[RecommendRankedOut],
    summary="뉴스 추천 랭킹",
    description="뉴스 추천 랭킹",
)
async def get_news_recommend(
    request: Request,
    payload: RecommendRankedIn,
    response: Response,
    db: Session = Depends(get_db),
):
    results = await get_news_recommended_ranked(payload, request, db)

    # Prometheus용 헤더 추가
    click_mean = np.mean([result["click_score"] for result in results[:5]])
    click_variance = np.var([result["click_score"] for result in results[:5]])

    response.headers["X-click-mean-score"] = f"{click_mean:.3f}"
    response.headers["X-click-variance-score"] = f"{click_variance:.6f}"

    return results
