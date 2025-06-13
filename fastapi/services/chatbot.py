from models.rag_pipeline import NewsTossChatbot

# 싱글턴 인스턴스 (서버 시작 시 1회만 로딩)
chatbot = NewsTossChatbot(
    llm_model_path="../models/quantized_llama3/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"
)

def get_chatbot_answer(question: str):
    return chatbot.answer(question)