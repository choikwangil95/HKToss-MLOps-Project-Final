from services.model_cache import get_model


def get_news_embedding(news_text: str):

    model = get_model()

    target_embedding = model.encode([news_text])[0]

    return target_embedding
