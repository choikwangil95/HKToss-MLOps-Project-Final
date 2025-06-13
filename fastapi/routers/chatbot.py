from fastapi import APIRouter, HTTPException
from schemas import ChatRequest, ChatResponse
from services.chatbot import get_chatbot_answer

router = APIRouter()

# def로 작성하면 FastAPI가 자동으로 스레드풀에서 병렬 실행
@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        result = get_chatbot_answer(request.question)
        return ChatResponse(answer=result["answer"], reranked_docs=result["reranked_docs"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
