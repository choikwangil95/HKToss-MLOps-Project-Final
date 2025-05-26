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

-- reports 테이블 생성
CREATE TABLE IF NOT EXISTS reports (
    report_id SERIAL PRIMARY KEY,
    stock_name TEXT,
    title TEXT,
    sec_firm TEXT,
    date DATE,
    view_count INT,
    url TEXT,
    target_price TEXT,
    opinion TEXT,
    report_content TEXT,
    embedding VECTOR(768)
);

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM reports) = 0 THEN
        COPY reports(stock_name, title, sec_firm, date, view_count, url, target_price, opinion, report_content, embedding)
        FROM '/docker-entrypoint-initdb.d/report_24_25_with_embeddings.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;
