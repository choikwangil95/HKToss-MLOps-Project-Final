from models.rag_pipeline import NewsTossChatbot

# 싱글턴 객체로 관리 (여러 요청에서 재사용)
chatbot = NewsTossChatbot()

def get_answer(question: str, top_k: int = 5) -> str:
    return chatbot.answer(question, top_k=top_k)