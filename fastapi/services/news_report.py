import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 불러오기
with open("./datas/reports(24_25)_embeddings.pkl", "rb") as f:
    report_embeddings = pickle.load(f)
report_meta_df = pd.read_csv("./datas/reports(24_25)_meta.csv")

with open("./datas/news(24_25)_embedding.pkl", "rb") as f:
    news_embeddings = pickle.load(f)

# 유사도 인덱스 → 메타 정보 같이 반환
def get_similar_past_reports(news_id: int, topk: int = 5):
    news_embedding = news_embeddings[news_id].reshape(1, -1)
    sims = cosine_similarity(news_embedding, report_embeddings)[0]
    top_indices = sims.argsort()[-topk:][::-1]

    results = []
    for idx in top_indices:
        row = report_meta_df.iloc[idx]
        results.append({
            "제목": row["제목"],
            "증권사": row["증권사"],
            "작성일": row["작성일"],
            "목표가": row["목표가"],
            "투자의견": row["투자의견"],
            "본문": row["본문"],
            "similarity": round(float(sims[idx]), 6)
        })
    return results

