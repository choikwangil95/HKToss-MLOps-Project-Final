import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# DB 연결
conn = psycopg2.connect(
    host='localhost',
    port='5432',
    dbname='postgres',
    user='postgres',
    password='password'
)
cursor = conn.cursor()

# 임베딩 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# embedding이 비어있는 news_id 가져오기 + summary 가져오기
cursor.execute("""
    SELECT news_id, summary
    FROM news_v2_metadata
    WHERE news_id IN (SELECT news_id FROM news_v2_embedding WHERE embedding IS NULL)
""")
rows = cursor.fetchall()

print(f'Found {len(rows)} rows to update.')

# 업데이트 루프 + batch commit
batch_size = 100
batch_counter = 0

for news_id, summary in tqdm(rows):
    embedding = model.encode(summary).tolist()

    cursor.execute("""
        UPDATE news_v2_embedding
        SET embedding = %s
        WHERE news_id = %s
    """, (embedding, news_id))

    batch_counter += 1
    if batch_counter % batch_size == 0:
        conn.commit()

# 마지막 남은 것 commit
conn.commit()

print('All embeddings updated.')

# 연결 종료
cursor.close()
conn.close()
