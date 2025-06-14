from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request, Body
from schemas.model import (
    EmbeddingIn,
    EmbeddingOut,
    StockOut,
    SummaryOut,
    SummaryIn,
    StockIn,
)
from services.model import (
    get_lda_topic,
    get_news_embedding,
    get_news_similar_list,
    get_news_summary,
    extract_ogg_economy,
    get_ner_tokens,
)
from db.label_map import id2label

router = APIRouter(
    prefix="/plm",
    tags=["Pre-Trained Model"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/summarize",
    response_model=SummaryOut,
    summary="뉴스 본문 요약",
    description="뉴스 본문을 입력받아 요약 결과 반환",
)
async def get_news_summary_router(request: Request, payload: SummaryIn):
    """
    뉴스 본문 요약
    """
    article = payload.article
    if not article:
        raise HTTPException(
            status_code=400,
            detail="기사 본문이 비어있습니다. 올바른 본문을 입력해주세요.",
        )

    summary = get_news_summary(article, request)  # ✅ await 제거

    return {"summary": summary}


@router.post(
    "/stocks",
    response_model=StockOut,
    summary="뉴스 본문 종목명 추출",
    description="뉴스 본문 종목명 추출",
)
async def get_stock_list_router(request: Request, payload: StockIn):
    """
    뉴스 본문 종목명 추출
    """
    article = payload.article
    if not article:
        raise HTTPException(
            status_code=400,
            detail="기사 본문이 비어있습니다. 올바른 본문을 입력해주세요.",
        )

    tokens, labels = get_ner_tokens(article, request, id2label)
    stock_list = extract_ogg_economy(tokens, labels)

    return {"stock_list": stock_list}


@router.post(
    "/embedding",
    response_model=EmbeddingOut,
    summary="뉴스 텍스트 임베딩",
    description="뉴스 텍스트 임베딩",
)
async def get_news_embedding_router(request: Request, payload: EmbeddingIn):
    """
    뉴스 텍스트 임베딩
    """
    article = payload.article
    if not article:
        raise HTTPException(
            status_code=400,
            detail="기사 본문이 비어있습니다. 올바른 본문을 입력해주세요.",
        )

    embedding = get_news_embedding(article, request)
    if embedding is None:
        raise HTTPException(
            status_code=500,
            detail="임베딩 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
        )

    return {"embedding": embedding}
