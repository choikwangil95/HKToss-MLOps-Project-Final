from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import os
import json

router = APIRouter()

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
