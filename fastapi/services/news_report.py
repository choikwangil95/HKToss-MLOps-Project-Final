import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# CSV 로드
news_df = pd.read_csv("./datas/news(24-25)_with_embeddings.csv")
report_df = pd.read_csv("./datas/report(24-25)_with_embeddings.csv")

# 임베딩 컬럼 파싱 함수
def parse_embedding(emb_str):
    # 문자열 -> list(float)
    if isinstance(emb_str, str):
        # 예: "[0.1, 0.2, 0.3]" 또는 '[-0.123, ...]'
        return np.array(json.loads(emb_str))
    else:
        return np.zeros(768)  # 예시: 임베딩 사이즈에 맞춰서 (오류방지)

# 전체 임베딩 배열 생성
news_embeddings = np.vstack(news_df["embedding"].apply(parse_embedding).to_list())
report_embeddings = np.vstack(report_df["embedding"].apply(parse_embedding).to_list())

def get_similar_past_reports(news_id: int, topk: int = 5):
    news_idx = int(news_id) - 1  # 0001 → 0
    news_embedding = news_embeddings[news_idx].reshape(1, -1)
    sims = cosine_similarity(news_embedding, report_embeddings)[0]
    top_indices = sims.argsort()[-topk:][::-1]

    results = []
    for idx in top_indices:
        row = report_df.iloc[idx]
        date_str = str(row["작성일"])
        try:
            report_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            report_date = date_str
        results.append({
            "report_title": row.get("report_title", row.get("제목", "")),
            "report_content": row.get("report_content", row.get("본문", "")),
            "report_date": report_date,
            "company": row.get("증권사", row.get("company", "")),
            "target_price": row.get("목표가", row.get("target_price", "")),
            "opinion": row.get("투자의견", row.get("opinion", "")),
            "similarity": round(float(sims[idx]), 6)
        })  

    return results