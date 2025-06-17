from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request, Body
from fastapi.models.news import NewsModel_v2, NewsModel_v2_External
from schemas.model import (
    ChatIn,
    ChatOut,
    LdaTopicsIn,
    LdaTopicsOut,
    SimilarNewsIn,
    SimilarNewsOut,
    SimilarityRequest,
    SimilarityResponse,
    SimilarityResult
)
from services.model import (
    get_lda_topic,
    get_news_similar_list,
    get_stream_response,
    compute_similarity
)

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import os
import json
from load_models import get_similarity_model
import asyncio
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
    "/similarity",
    response_model=SimilarityResponse,
    summary="회귀 모델 기반 뉴스 유사도 예측",
    description="기준 뉴스 ID를 기반으로 회귀 모델이 예측한 유사도 상위 5개 뉴스 결과를 반환합니다."
)
async def get_similarity_scores(request: Request, payload: SimilarityRequest, db: Session = Depends(get_db)):
    # 로드된 모델 가져오기
    # scalers = request.app.state.scalers
    
    # news_id = payload.news_id
    # news_topk_ids = payload.news_topk_ids or []

    # # 기준 뉴스 정보 조회
    # ref_news_raw = (
    #     db.query(NewsModel_v2).filter(NewsModel_v2.news_id == news_id).first()
    # )
    # print(ref_news_raw)
    # summary = ref_news_raw.summary if ref_news_raw else None

    # ref_news_external = (
    #     db.query(NewsModel_v2_External).filter(NewsModel_v2_External.news_id == news_id).first()
    # )
    # extA = target.iloc[0][[col for col in df_meta.columns if col.startswith('extA_')]].tolist()

    # 토픽임
    # ref_news_external = (
    #     db.query(NewsModel_v2_External).filter(NewsModel_v2_External.news_id == news_id).first()
    # )

    # 2. 유사 뉴스 검색
    # topk_news_raw = (
    #     db.query(NewsModel_v2).filter(NewsModel_v2.news_id.in_(news_topk_ids)).all()
    # )
    # topk_news_ids = [news.news_id for news in topk_news_raw]

    # topk_news_external = (
    #     db.query(NewsModel_v2_External).filter(NewsModel_v2_External.news_id.in_(news_topk_ids)).all()
    # )


    # # 2. 유사 뉴스 검색 (동기 코드 → run_in_executor)

    # # 유사 뉴스들의 외부 변수와 summary 가져오기
    # similar_df = df_meta[df_meta['news_id'].isin(similar_ids)]
    # similar_extBs = similar_df[[col for col in df_meta.columns if col.startswith('extB_similar_')]].values.tolist()
    # similar_summaries = similar_df['summary'].tolist()

    # # 텍스트 임베딩 API 함수 정의 (현재는 임시로 직접 넣는 구조)
    # def embedding_api_func(texts):
    #     return request.app.state.embedding_model.encode(texts) 

    # # 유사도 점수 계산
    # results = compute_similarity(
    #     summary,
    #     similar_summaries,
    #     extA,
    #     similar_extBs,
    #     scalers,
    #     ae_sess,
    #     regressor_sess,
    #     tokenizer,
    #     embedding_api_func
    # )

    # # 뉴스 ID 매핑
    # news_id_map = dict(zip(similar_summaries, similar_ids))
    # for r in results:
    #     r['news_id'] = news_id_map.get(r['summary'], 'unknown')

    # return {"results": [SimilarityResult(**r) for r in results]}
    pass