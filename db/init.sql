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
  embedding VECTOR(768),
  stocks TEXT  -- 주식 종목을 저장하기 위한 컬럼
);

-- 테이블에 데이터가 없을 때만 CSV에서 데이터 COPY 수행
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news) = 0 THEN
        COPY news(news_id, date, title, url, content, embedding, stocks)
        FROM '/docker-entrypoint-initdb.d/news_3y.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;


-- news_v2 테이블 생성
CREATE TABLE news_v2 (
  news_id VARCHAR PRIMARY KEY,
  wdate TIMESTAMP,
  title TEXT,
  article TEXT,
  press TEXT,
  url TEXT,
  image TEXT
);

-- 테이블에 데이터가 없을 때만 CSV에서 데이터 COPY 수행
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news_v2) = 0 THEN
        COPY news_v2(news_id, wdate, title, article, press, url, image)
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_raw.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;


-- news_v2_metadata 테이블 생성 (news_id에 외래키 제약조건 추가)
CREATE TABLE news_v2_metadata (
  news_id VARCHAR PRIMARY KEY,
  summary TEXT,
  stock_list JSON,
  industry_list JSON,
  CONSTRAINT fk_news_id FOREIGN KEY (news_id) REFERENCES news_v2(news_id) ON DELETE CASCADE
);

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news_v2_metadata) = 0 THEN
        COPY news_v2_metadata(news_id, summary, stock_list, industry_list)
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_metadata.csv'
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
