-- news DATABASE 생성
CREATE DATABASE news_db;

-- pgvector 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;

-- news 테이블 생성
CREATE TABLE news (
  news_id VARCHAR PRIMARY KEY,
  date DATE,
  title TEXT,
  url TEXT,
  content TEXT,
  embedding VECTOR(768)
);

-- 테이블에 데이터가 없을 때만 CSV에서 데이터 COPY 수행
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news) = 0 THEN
        COPY news(news_id, date, title, url, content, embedding)
        FROM '/docker-entrypoint-initdb.d/news_with_embedding.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;
