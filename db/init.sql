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
  impact_score FLOAT,
  CONSTRAINT fk_news_id FOREIGN KEY (news_id) REFERENCES news_v2(news_id) ON DELETE CASCADE
);

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news_v2_metadata) = 0 THEN
        COPY news_v2_metadata(news_id, summary, stock_list, industry_list, impact_score)
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_metadata.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;

-- news_v2_external 테이블 생성 (news_id에 외래키 제약조건 추가)
CREATE TABLE news_v2_external (
  news_id VARCHAR PRIMARY KEY,

  d_minus_14_date_close FLOAT,
  d_minus_14_date_volume FLOAT,
  d_minus_14_date_foreign FLOAT,
  d_minus_14_date_institution FLOAT,
  d_minus_14_date_individual FLOAT,

  d_minus_7_date_close FLOAT,
  d_minus_7_date_volume FLOAT,
  d_minus_7_date_foreign FLOAT,
  d_minus_7_date_institution FLOAT,
  d_minus_7_date_individual FLOAT,

  d_minus_3_date_close FLOAT,
  d_minus_3_date_volume FLOAT,
  d_minus_3_date_foreign FLOAT,
  d_minus_3_date_institution FLOAT,
  d_minus_3_date_individual FLOAT,

  d_minus_2_date_close FLOAT,
  d_minus_2_date_volume FLOAT,
  d_minus_2_date_foreign FLOAT,
  d_minus_2_date_institution FLOAT,
  d_minus_2_date_individual FLOAT,

  d_minus_1_date_close FLOAT,
  d_minus_1_date_volume FLOAT,
  d_minus_1_date_foreign FLOAT,
  d_minus_1_date_institution FLOAT,
  d_minus_1_date_individual FLOAT,

  d_day_date_close FLOAT,
  d_day_date_volume FLOAT,
  d_day_date_foreign FLOAT,
  d_day_date_institution FLOAT,
  d_day_date_individual FLOAT,

  d_plus_1_date_close FLOAT,
  d_plus_1_date_volume FLOAT,
  d_plus_1_date_foreign FLOAT,
  d_plus_1_date_institution FLOAT,
  d_plus_1_date_individual FLOAT,

  d_plus_2_date_close FLOAT,
  d_plus_2_date_volume FLOAT,
  d_plus_2_date_foreign FLOAT,
  d_plus_2_date_institution FLOAT,
  d_plus_2_date_individual FLOAT,

  d_plus_3_date_close FLOAT,
  d_plus_3_date_volume FLOAT,
  d_plus_3_date_foreign FLOAT,
  d_plus_3_date_institution FLOAT,
  d_plus_3_date_individual FLOAT,

  d_plus_7_date_close FLOAT,
  d_plus_7_date_volume FLOAT,
  d_plus_7_date_foreign FLOAT,
  d_plus_7_date_institution FLOAT,
  d_plus_7_date_individual FLOAT,

  d_plus_14_date_close FLOAT,
  d_plus_14_date_volume FLOAT,
  d_plus_14_date_foreign FLOAT,
  d_plus_14_date_institution FLOAT,
  d_plus_14_date_individual FLOAT,

  d_day_date_open FLOAT,
  d_day_change_open FLOAT,
  d_day_change FLOAT,

  fx FLOAT,
  bond10y FLOAT,
  base_rate FLOAT,

  CONSTRAINT fk_news_id FOREIGN KEY (news_id) REFERENCES news_v2(news_id) ON DELETE CASCADE
);



DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news_v2_external) = 0 THEN
        COPY news_v2_external (
            news_id,
            d_minus_14_date_close, d_minus_14_date_volume, d_minus_14_date_foreign, d_minus_14_date_institution, d_minus_14_date_individual,
            d_minus_7_date_close, d_minus_7_date_volume, d_minus_7_date_foreign, d_minus_7_date_institution, d_minus_7_date_individual,
            d_minus_3_date_close, d_minus_3_date_volume, d_minus_3_date_foreign, d_minus_3_date_institution, d_minus_3_date_individual,
            d_minus_2_date_close, d_minus_2_date_volume, d_minus_2_date_foreign, d_minus_2_date_institution, d_minus_2_date_individual,
            d_minus_1_date_close, d_minus_1_date_volume, d_minus_1_date_foreign, d_minus_1_date_institution, d_minus_1_date_individual,
            d_day_date_close, d_day_date_volume, d_day_date_foreign, d_day_date_institution, d_day_date_individual,
            d_plus_1_date_close, d_plus_1_date_volume, d_plus_1_date_foreign, d_plus_1_date_institution, d_plus_1_date_individual,
            d_plus_2_date_close, d_plus_2_date_volume, d_plus_2_date_foreign, d_plus_2_date_institution, d_plus_2_date_individual,
            d_plus_3_date_close, d_plus_3_date_volume, d_plus_3_date_foreign, d_plus_3_date_institution, d_plus_3_date_individual,
            d_plus_7_date_close, d_plus_7_date_volume, d_plus_7_date_foreign, d_plus_7_date_institution, d_plus_7_date_individual,
            d_plus_14_date_close, d_plus_14_date_volume, d_plus_14_date_foreign, d_plus_14_date_institution, d_plus_14_date_individual,
            d_day_date_open, d_day_change_open, d_day_change,
            fx, bond10y, base_rate
        )
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_external.csv'
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
