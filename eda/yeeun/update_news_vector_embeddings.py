import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# DB 연결 (EC2 서버 연결 정보로 수정 필요)
conn = psycopg2.connect(
    host='3.37.207.16',
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
    SELECT v.news_id, m.summary
    FROM news_v2_embedding v
    JOIN news_v2_metadata m ON v.news_id = m.news_id
    WHERE v.embedding IS NULL AND m.summary IS NOT NULL
""")
rows = cursor.fetchall()

print(f'Found {len(rows)} rows to update.')

# 업데이트 루프 + batch commit
batch_size = 100
batch_counter = 0

for news_id, summary in tqdm(rows):
    embedding = model.encode(summary).tolist()

    # vector 형식으로 변환 → 반드시 [] 로 해야 오류 안 남
    embedding_pg_vector = "[" + ",".join([str(x) for x in embedding]) + "]"

    cursor.execute("""
        UPDATE news_v2_embedding
        SET embedding = %s::vector
        WHERE news_id = %s
    """, (embedding_pg_vector, news_id))

    batch_counter += 1
    if batch_counter % batch_size == 0:
        conn.commit()

# 마지막 남은 것 commit
conn.commit()

print('All embeddings updated.')

# 연결 종료
cursor.close()
conn.close()