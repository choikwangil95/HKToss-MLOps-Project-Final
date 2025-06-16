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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import os
import json

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


@router.post(
    "/similar_news",
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

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

@router.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    # 1. RAG 파이프라인으로 context, prompt 생성 (동기 함수라면 run_in_executor 사용)
    from models.rag_pipeline import NewsTossChatbot
    chatbot = NewsTossChatbot()

    import asyncio
    loop = asyncio.get_event_loop()
    # prompt 생성은 동기 함수라면 run_in_executor로 감싸기
    prompt = await loop.run_in_executor(None, chatbot.make_stream_prompt, req.question, req.top_k)

    # 2. OpenAI 스트리밍 호출
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1024,
        stream=True,  # 스트리밍 활성화!
    )

    async def event_stream():
        # OpenAI의 stream은 동기 iterator이므로, run_in_executor로 감싸야 함
        def sync_stream():
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield f"data: {content}\n\n"
        # 비동기 제너레이터로 변환
        for item in await loop.run_in_executor(None, lambda: list(sync_stream())):
            yield item
    
    # SSE(Server-Sent Events)
    return StreamingResponse(event_stream(), media_type="text/event-stream") 
