from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request, Body
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


from pydantic import BaseModel
import openai
import os
import json
from load_models import get_similarity_model
import asyncio
from sqlalchemy.orm import Session
from db.postgresql import get_db
from models.custom import NewsModel_v2, NewsModel_v2_External, NewsModel_v2_Topic


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
    scalers = request.app.state.scalers
    ae_sess = request.app.state.ae_sess
    regressor_sess = request.app.state.regressor_sess
    embedding_api_func = lambda texts: requests.post(
        request.app.state.embedding_api_url,
        json={'texts': texts}
    ).json()['embeddings']  # 임베딩 API로 POST 요청

    news_id = payload.news_id
    news_topk_ids = payload.news_topk_ids or []

    # 기준 뉴스 정보 조회
    ref_news_raw = db.query(NewsModel_v2).filter(NewsModel_v2.news_id == news_id).first()
    if not ref_news_raw:
        raise HTTPException(status_code=404, detail="기준 뉴스 정보를 찾을 수 없습니다.")
    summary = ref_news_raw.summary

    ref_news_external = db.query(NewsModel_v2_External).filter(NewsModel_v2_External.news_id == news_id).first()
    if not ref_news_external:
        raise HTTPException(status_code=404, detail="기준 뉴스 외부 변수 정보를 찾을 수 없습니다.")

    extA_cols = [col.name for col in ref_news_external.__table__.columns if col.name.startswith('extA_')]
    extA = [getattr(ref_news_external, col) for col in extA_cols]

    ref_news_topic = db.query(NewsModel_v2_Topic).filter(NewsModel_v2_Topic.news_id == news_id).first()
    if not ref_news_topic:
        raise HTTPException(status_code=404, detail="기준 뉴스 토픽 정보를 찾을 수 없습니다.")

    topic_cols = [col.name for col in ref_news_topic.__table__.columns if col.name.startswith('topic_')]
    topicA = [getattr(ref_news_topic, col) for col in topic_cols]

    extA_total = extA + topicA

    # 유사 뉴스 정보 조회
    topk_news_raw = db.query(NewsModel_v2).filter(NewsModel_v2.news_id.in_(news_topk_ids)).all()
    summary_map = {news.news_id: news.summary for news in topk_news_raw}
    try:
        similar_summaries = [summary_map[nid] for nid in news_topk_ids]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"유사 뉴스 ID {str(e)}가 DB에 존재하지 않습니다.")

    topk_exts = db.query(NewsModel_v2_External).filter(NewsModel_v2_External.news_id.in_(news_topk_ids)).all()
    ext_map = {ext.news_id: ext for ext in topk_exts}

    extB_cols = [col.name for col in NewsModel_v2_External.__table__.columns if col.name.startswith('extB_similar_')]
    extBs = [
        [getattr(ext_map[nid], col) for col in extB_cols]
        for nid in news_topk_ids
    ]

    topk_topics = db.query(NewsModel_v2_Topic).filter(NewsModel_v2_Topic.news_id.in_(news_topk_ids)).all()
    topic_map = {topic.news_id: topic for topic in topk_topics}

    topicB_cols = [col.name for col in NewsModel_v2_Topic.__table__.columns if col.name.startswith('similar_topic_')]
    topicBs = [
        [getattr(topic_map[nid], col) for col in topicB_cols]
        for nid in news_topk_ids
    ]

    extB_total = [
        ext + topic for ext, topic in zip(extBs, topicBs)
    ]

    missing_ext_ids = [nid for nid in news_topk_ids if nid not in ext_map]
    missing_topic_ids = [nid for nid in news_topk_ids if nid not in topic_map]

    if missing_ext_ids:
        raise HTTPException(status_code=400, detail=f"외부 변수 없는 뉴스 ID: {missing_ext_ids}")
    if missing_topic_ids:
        raise HTTPException(status_code=400, detail=f"토픽 변수 없는 뉴스 ID: {missing_topic_ids}")

    # 유사도 점수 계산
    results = compute_similarity(
         summary,
         similar_summaries,
         extA_total,
         extB_total,
         scalers,
         ae_sess,
         regressor_sess,
         embedding_api_func
    )

    # news_id 매핑
    news_id_map = dict(zip(similar_summaries, news_topk_ids))

    for r in results:
        r['news_id'] = news_id_map.get(r['summary'], 'unknown')

    return SimilarityResponse(results=[SimilarityResult(**r) for r in results])