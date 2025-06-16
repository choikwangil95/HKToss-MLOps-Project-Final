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