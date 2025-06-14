from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request, Body
from schemas.model import (
    LdaTopicsIn,
    LdaTopicsOut,
    SimilarNewsIn,
    SimilarNewsOut,
)
from services.model import (
    get_lda_topic,
    get_news_similar_list,
)

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
