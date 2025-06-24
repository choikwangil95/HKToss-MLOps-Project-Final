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
  stock_list_view JSON,
  industry_list JSON,
  impact_score FLOAT,
  CONSTRAINT fk_news_id FOREIGN KEY (news_id) REFERENCES news_v2(news_id) ON DELETE CASCADE
);

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news_v2_metadata) = 0 THEN
        COPY news_v2_metadata(news_id, summary, stock_list, stock_list_view, industry_list, impact_score)
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_metadata2.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;

-- news_v2_external 테이블 생성 (news_id에 외래키 제약조건 추가)
CREATE TABLE news_v2_external (
  news_id VARCHAR PRIMARY KEY,

  d_minus_5_date_close FLOAT,
  d_minus_5_date_volume FLOAT,
  d_minus_5_date_foreign FLOAT,
  d_minus_5_date_institution FLOAT,
  d_minus_5_date_individual FLOAT,

  d_minus_4_date_close FLOAT,
  d_minus_4_date_volume FLOAT,
  d_minus_4_date_foreign FLOAT,
  d_minus_4_date_institution FLOAT,
  d_minus_4_date_individual FLOAT,

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

  d_plus_1_date_close FLOAT,
  d_plus_2_date_close FLOAT,
  d_plus_3_date_close FLOAT,
  d_plus_4_date_close FLOAT,
  d_plus_5_date_close FLOAT,

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
            d_minus_5_date_close, d_minus_5_date_volume, d_minus_5_date_foreign, d_minus_5_date_institution, d_minus_5_date_individual,
            d_minus_4_date_close, d_minus_4_date_volume, d_minus_4_date_foreign, d_minus_4_date_institution, d_minus_4_date_individual,
            d_minus_3_date_close, d_minus_3_date_volume, d_minus_3_date_foreign, d_minus_3_date_institution, d_minus_3_date_individual,
            d_minus_2_date_close, d_minus_2_date_volume, d_minus_2_date_foreign, d_minus_2_date_institution, d_minus_2_date_individual,
            d_minus_1_date_close, d_minus_1_date_volume, d_minus_1_date_foreign, d_minus_1_date_institution, d_minus_1_date_individual,
            d_plus_1_date_close, d_plus_2_date_close, d_plus_3_date_close, d_plus_4_date_close, d_plus_5_date_close,
            fx, bond10y, base_rate
        )
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_external.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;


-- news_v2 테이블 생성
CREATE TABLE news_v2_topic (
  news_id VARCHAR PRIMARY KEY,
  topic_1 FLOAT,
  topic_2 FLOAT,
  topic_3 FLOAT,
  topic_4 FLOAT,
  topic_5 FLOAT,
  topic_6 FLOAT,
  topic_7 FLOAT,
  topic_8 FLOAT,
  topic_9 FLOAT
);

-- 테이블에 데이터가 없을 때만 CSV에서 데이터 COPY 수행
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news_v2_topic) = 0 THEN
        COPY news_v2_topic(news_id, topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7, topic_8, topic_9)
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_tm.csv'
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

-- DO $$
-- BEGIN
--     IF (SELECT COUNT(*) FROM reports) = 0 THEN
--        COPY reports(stock_name, title, sec_firm, date, view_count, url, target_price, opinion, report_content, embedding)
--        FROM '/docker-entrypoint-initdb.d/report_24_25_with_embeddings.csv'
--        WITH (FORMAT csv, HEADER true);
--    END IF;
-- END $$;


-- news_v2_embedding 테이블 생성 (그대로 유지)
CREATE TABLE IF NOT EXISTS news_v2_embedding (
    news_id VARCHAR PRIMARY KEY,
    wdate TIMESTAMP,
    embedding VECTOR(768),
    CONSTRAINT fk_news_id FOREIGN KEY (news_id) REFERENCES news_v2(news_id) ON DELETE CASCADE
);

-- news_id 먼저 COPY → news_2023_2025_metadata.csv 사용
-- COPY news_v2_embedding(news_id)
-- FROM '/docker-entrypoint-initdb.d/news_id_only.csv'
-- WITH (FORMAT csv, HEADER true);

-- wdate 업데이트
-- UPDATE news_v2_embedding v
-- SET wdate = n.wdate
-- FROM news_v2 n
-- WHERE v.news_id = n.news_id;

-- embedding 업데이트 (Python에서 summary 임베딩 후)
-- UPDATE news_v2_embedding
-- SET embedding = '여기에 VECTOR 문자열'::VECTOR
-- WHERE news_id = '특정 뉴스 ID';

-- news_v2 테이블 생성
CREATE TABLE IF NOT EXISTS news_v2_similar (
  news_id VARCHAR NOT NULL,
  sim_news_id VARCHAR NOT NULL,
  wdate TIMESTAMP,
  title TEXT,
  summary TEXT,
  press TEXT,
  url TEXT,
  image TEXT,
  similarity FLOAT,

  PRIMARY KEY (news_id, sim_news_id),
  
  CONSTRAINT fk_news_id FOREIGN KEY (news_id) REFERENCES news_v2(news_id) ON DELETE CASCADE,
  CONSTRAINT fk_sim_news_id FOREIGN KEY (sim_news_id) REFERENCES news_v2(news_id) ON DELETE CASCADE
);

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news_v2_similar) = 0 THEN
        COPY news_v2_similar(news_id, sim_news_id, wdate, title, summary, press, url, image, similarity)
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_similarity2.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;


-- news_v2_log 테이블 생성
CREATE TABLE news_v2_log (
  id INT,
  user_id TEXT,
  news_id TEXT,
  wdate TIMESTAMP
);

-- 테이블에 데이터가 없을 때만 CSV에서 데이터 COPY 수행
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM news_v2_log) = 0 THEN
        COPY news_v2_log(id, user_id, news_id, wdate)
        FROM '/docker-entrypoint-initdb.d/news_2023_2025_log.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;


-- user_profile 테이블 생성 
CREATE TABLE user_profile (
  user_id VARCHAR PRIMARY KEY,
  userPnl INT,
  asset	INT,
  investScore INT,
  memberStocks JSON
);

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM user_profile) = 0 THEN
        COPY user_profile(user_id, userPnl, asset, investScore, memberStocks)
        FROM '/docker-entrypoint-initdb.d/user_profile.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;
