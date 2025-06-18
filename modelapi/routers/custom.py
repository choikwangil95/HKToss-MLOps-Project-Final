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
    SimilarityRequest,
    SimilarityResponse,
    SimilarityResult,
)
from services.model import (
    get_lda_topic,
    get_news_embeddings,
    get_news_recommended,
    get_news_similar_list,
    get_stream_response,
    compute_similarity,
)
import requests

from services.custom import get_news_impact_score_service

from schemas.custom import SimpleImpactResponse

import numpy as np

from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import openai
import os
import json
from load_models import get_similarity_model
import asyncio
from sqlalchemy.orm import Session
from db.postgresql import get_db
from models.custom import (
    NewsModel_v2_Metadata,
    NewsModel_v2_External,
    NewsModel_v2_Topic,
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
    description="기준 뉴스 ID를 기반으로 회귀 모델이 예측한 유사도 상위 5개 뉴스 결과를 반환합니다.",
)
async def get_similarity_scores(
    request: Request, payload: SimilarityRequest, db: Session = Depends(get_db)
):
    # 로드된 모델 가져오기
    scalers = request.app.state.scalers
    ae_sess = request.app.state.ae_sess
    regressor_sess = request.app.state.regressor_sess

    async def embedding_api_func(texts):
        embeddings = await get_news_embeddings(texts, request)

        print("🟡 임베딩 결과:", embeddings)

        return embeddings

    news_id = payload.news_id
    news_topk_ids = payload.news_topk_ids or []

    # 공통 외부변수 컬럼 정의 (news_id 제외 전부)
    ext_cols = [
        col.name
        for col in NewsModel_v2_External.__table__.columns
        if col.name != "news_id"
    ]

    # 기준 뉴스 정보 조회
    ref_news_raw = (
        db.query(NewsModel_v2_Metadata)
        .filter(NewsModel_v2_Metadata.news_id == news_id)
        .first()
    )
    if not ref_news_raw:
        raise HTTPException(
            status_code=404, detail="기준 뉴스 정보를 찾을 수 없습니다."
        )
    summary = ref_news_raw.summary

    ref_news_external = (
        db.query(NewsModel_v2_External)
        .filter(NewsModel_v2_External.news_id == news_id)
        .first()
    )
    if not ref_news_external:
        raise HTTPException(
            status_code=404, detail="기준 뉴스 외부 변수 정보를 찾을 수 없습니다."
        )
    extA = [getattr(ref_news_external, col) for col in ext_cols]

    ref_news_topic = (
        db.query(NewsModel_v2_Topic)
        .filter(NewsModel_v2_Topic.news_id == news_id)
        .first()
    )
    if not ref_news_topic:
        raise HTTPException(
            status_code=404, detail="기준 뉴스 토픽 정보를 찾을 수 없습니다."
        )
    topic_cols = [
        col.name
        for col in ref_news_topic.__table__.columns
        if col.name.startswith("topic_")
    ]
    topicA = [getattr(ref_news_topic, col) for col in topic_cols]

    extA_total = extA + topicA

    # 유사 뉴스 정보 조회
    topk_news_raw = (
        db.query(NewsModel_v2_Metadata)
        .filter(NewsModel_v2_Metadata.news_id.in_(news_topk_ids))
        .all()
    )
    summary_map = {news.news_id: news.summary for news in topk_news_raw}
    try:
        similar_summaries = [summary_map[nid] for nid in news_topk_ids]
    except KeyError as e:
        raise HTTPException(
            status_code=400, detail=f"유사 뉴스 ID {str(e)}가 DB에 존재하지 않습니다."
        )

    topk_exts = (
        db.query(NewsModel_v2_External)
        .filter(NewsModel_v2_External.news_id.in_(news_topk_ids))
        .all()
    )
    ext_map = {ext.news_id: ext for ext in topk_exts}
    extBs = [[getattr(ext_map[nid], col) for col in ext_cols] for nid in news_topk_ids]

    topk_topics = (
        db.query(NewsModel_v2_Topic)
        .filter(NewsModel_v2_Topic.news_id.in_(news_topk_ids))
        .all()
    )
    topic_map = {topic.news_id: topic for topic in topk_topics}
    topicB_cols = [
        col.name
        for col in NewsModel_v2_Topic.__table__.columns
        if col.name.startswith("topic_")
    ]
    topicBs = [
        [getattr(topic_map[nid], col) for col in topicB_cols] for nid in news_topk_ids
    ]

    extB_total = [ext + topic for ext, topic in zip(extBs, topicBs)]

    missing_ext_ids = [nid for nid in news_topk_ids if nid not in ext_map]
    missing_topic_ids = [nid for nid in news_topk_ids if nid not in topic_map]

    if missing_ext_ids:
        raise HTTPException(
            status_code=400, detail=f"외부 변수 없는 뉴스 ID: {missing_ext_ids}"
        )
    if missing_topic_ids:
        raise HTTPException(
            status_code=400, detail=f"토픽 변수 없는 뉴스 ID: {missing_topic_ids}"
        )

    # 유사도 점수 계산
    results = await compute_similarity(
        db=db,
        summary=summary,
        extA=extA,
        topicA=topicA,
        similar_summaries=similar_summaries,
        extBs=extBs,
        topicBs=topicBs,
        scalers=scalers,
        ae_sess=ae_sess,
        regressor_sess=regressor_sess,
        embedding_api_func=embedding_api_func,
        ext_col_names=ext_cols,
        topic_col_names=topic_cols,
        news_topk_ids=news_topk_ids,
    )

    # news_id 매핑
    news_id_map = dict(zip(similar_summaries, news_topk_ids))
    for r in results:
        r["news_id"] = news_id_map.get(r["summary"], "unknown")

    # 유사도 score 기준 정렬
    results.sort(key=lambda x: x["score"], reverse=True)

    return SimilarityResponse(results=[SimilarityResult(**r) for r in results])


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
