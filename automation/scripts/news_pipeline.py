# news_pipeline.py
import requests, random, time, os, logging, concurrent.futures
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from requests.adapters import HTTPAdapter, Retry
from logging.handlers import RotatingFileHandler
import psycopg2
from psycopg2.extras import execute_batch
import re
from kss import split_sentences
import os
import pandas as pd
import re
from konlpy.tag import Okt
import numpy as np
import json
import redis

# ──────────────────────────────
# 📌 로그 설정
# ──────────────────────────────


log = logging.getLogger("news_logger")
log.setLevel(logging.INFO)

# 로그 폴더 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 🔧 포맷터 설정
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# 🔧 파일 핸들러 설정
log_path = os.path.join(LOG_DIR, "news.log")
file_handler = RotatingFileHandler(
    log_path,
    maxBytes=5 * 1024 * 1024,  # 5MB 넘으면 순환
    backupCount=3,  # 최대 3개 백업 보관
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

# 🔧 콘솔 출력도 원할 경우 (선택)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


# ──────────────────────────────
# 📌 유틸 함수
# ──────────────────────────────

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LAST_CRAWLED_FILE = os.path.join(DATA_DIR, "last_crawled.txt")


def parse_wdate(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


def convert_to_public_url(href):
    parsed = urlparse(href)
    params = parse_qs(parsed.query)
    article_id = params.get("article_id", [""])[0]
    office_id = params.get("office_id", [""])[0]
    if article_id and office_id:
        return f"https://n.news.naver.com/mnews/article/{office_id}/{article_id}"
    return href


def get_random_headers():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1",
    ]
    return {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }


def safe_soup_parse(text, timeout=3):
    def parse():
        return BeautifulSoup(text, "lxml")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(parse)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.error("soup 파싱 타임아웃 발생")
            return None
        except Exception as e:
            log.error(f"soup 파싱 실패: {type(e).__name__}: {e}")
            return None


def get_retry_session():
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_article_details(url):
    try:
        headers = get_random_headers()
        # log.info(f"📰 요청 URL: {url}")

        session = get_retry_session()
        res = session.get(url, headers=headers, timeout=10)
        # log.info(f"📅 응답 상태 코드: {res.status_code}")
        res.raise_for_status()
        # log.info(f"📄 응답 본문 길이: {len(res.text)}")

        soup = safe_soup_parse(res.text)
        if soup is None:
            return None, ""

        # log.info("soup 생성 완료")

        image_tag = soup.select_one('meta[property="og:image"]')
        image = (
            image_tag["content"]
            if image_tag and image_tag.has_attr("content")
            else None
        )

        article_tag = soup.select_one("article#dic_area")
        article = (
            article_tag.get_text(strip=True, separator="\n") if article_tag else ""
        )

        # log.info(f"추출 성공: 이미지 있음? {bool(image)}, 본문 길이: {len(article)}")

        # 요청 사이에 무작위 대기
        time.sleep(random.uniform(1.0, 2.5))

        return image, article

    except Exception as e:
        log.error(f"전체 fetch 실패 - {url}: {type(e).__name__}: {e}")
        return None, ""


def get_or_create_last_time(filepath: str) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                last_time = f.read().strip()
            if last_time:
                log.info(f"이전 기록된 시간: {last_time}")
                return last_time
        log.info("기록이 없어 현재 시각으로 초기화")
        with open(filepath, "w") as f:
            f.write(now_str)
        return now_str
    except Exception as e:
        log.error(f"시간 읽기/쓰기 실패 ({type(e).__name__}): {e}")
        return now_str


def save_latest_time(filepath: str, time_str: str):
    try:
        with open(filepath, "w") as f:
            f.write(time_str)
        log.info(f"최신 시간 저장: {time_str}")
    except Exception as e:
        log.error(f"시간 저장 실패 ({type(e).__name__}): {e}")


# 중복 방지용 세트
generated_ids = set()


def generate_news_id(date_str):
    while True:
        rand_part = f"{random.randint(0, 99999999):08d}"  # 0000 ~ 9999
        news_id = f"{date_str}_{rand_part}"
        if news_id not in generated_ids:
            generated_ids.add(news_id)
            return news_id


def save_to_db(articles):
    if not articles:
        log.info("저장할 뉴스 없음")
        return

    try:
        DB_URL = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/news_db"
        )
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        insert_query = """
		INSERT INTO news_v2 (news_id, wdate, title, article, press, url, image)
		VALUES (%s, %s, %s, %s, %s, %s, %s)
		ON CONFLICT (news_id) DO NOTHING;
		"""

        values = [
            (
                article["news_id"],
                parse_wdate(article["wdate"]),
                article["title"],
                article["article"],
                article["press"],
                article["url"],
                article["image"],
            )
            for article in articles
        ]

        execute_batch(cur, insert_query, values)
        conn.commit()

        log.info(f"🧾 DB 저장 완료: {len(values)}건 저장")

    except Exception as e:
        log.error(f"❌ DB 저장 중 오류 ({type(e).__name__}): {e}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def save_to_db_metadata(articles):
    if not articles:
        log.info("저장할 뉴스 없음")
        return

    try:
        DB_URL = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/news_db"
        )
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        insert_query = """
		INSERT INTO news_v2_metadata (news_id, summary, stock_list, industry_list, impact_score)
		VALUES (%s, %s, %s, %s, %s)
		ON CONFLICT (news_id) DO NOTHING;
		"""

        print(f"==================={articles}===================")

        values = [
            (
                article["news_id"],
                article["summary"],
                (
                    json.dumps(article["stock_list"])
                    if article["stock_list"] is not None
                    else None
                ),
                (
                    json.dumps(article["industry_list"])
                    if article["industry_list"] is not None
                    else None
                ),
                None,  # impact_score는 아직 계산되지 않음
            )
            for article in articles
        ]

        execute_batch(cur, insert_query, values)
        conn.commit()

        log.info(f"🧾 Metadata DB 저장 완료: {len(values)}건 저장")

    except Exception as e:
        log.error(f"❌ Metadata DB 저장 중 오류 ({type(e).__name__}): {e}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# ──────────────────────────────
# 📌 뉴스 수집 메인 함수
# ──────────────────────────────


def fetch_latest_news():
    NEWS_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402"

    try:
        res = requests.get(NEWS_URL, headers=get_random_headers(), timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
    except Exception as e:
        log.error(f"뉴스 목록 요청 실패 ({type(e).__name__}): {e}")
        return []

    last_time = get_or_create_last_time(LAST_CRAWLED_FILE)
    last_time_dt = parse_wdate(last_time)

    articles = soup.select("dl > dd.articleSummary")
    new_articles = []

    for article in articles:
        try:
            title_tag = article.find_previous_sibling("dd", class_="articleSubject").a
            title = title_tag.text.strip()
            url = convert_to_public_url(title_tag["href"])
            press = article.select_one(".press").text.strip()
            wdate = article.select_one(".wdate").text.strip()
            article_time_dt = parse_wdate(wdate)

            if article_time_dt > last_time_dt:
                new_articles.append(
                    {
                        "wdate": wdate,
                        "title": title,
                        "press": press,
                        "url": url,
                    }
                )
        except Exception as e:
            log.error(f"개별 뉴스 파싱 실패 ({type(e).__name__}): {e}")

    log.info(f"새 뉴스 수: {len(new_articles)}")

    new_articles_crawled = []

    if new_articles:
        latest_time = max(parse_wdate(a["wdate"]) for a in new_articles)
        save_latest_time(LAST_CRAWLED_FILE, latest_time.strftime("%Y-%m-%d %H:%M"))

        for article in new_articles:
            try:
                # log.info(f"\n기사 처리 중: {article['title']}")
                image, article_text = fetch_article_details(article["url"])

                wdate = article["wdate"]
                title = article["title"]
                press = article["press"]
                url = article["url"]

                article_time_dt = parse_wdate(wdate)
                date_str = article_time_dt.strftime("%Y%m%d")

                news_id = generate_news_id(date_str)

                new_articles_crawled.append(
                    {
                        "news_id": news_id,
                        "wdate": wdate,
                        "title": title,
                        "article": article_text,
                        "press": press,
                        "url": url,
                        "image": image,
                    }
                )

                preview = article_text[:300] if isinstance(article_text, str) else ""
                log.info(
                    f"[NEW] {article['wdate']} - {article['title']} ({article['press']})"
                )
                log.info(f"{preview}...\n")
            except Exception as e:
                log.error(f"본문 파싱 실패 ({type(e).__name__}): {e}")
    else:
        log.info("새 뉴스 없음")

    return new_articles_crawled


# ──────────────────────────────
# 📌 뉴스 수집 메인 함수
# ──────────────────────────────


def remove_market_related_sentences(text: str) -> str:
    # 줄바꿈 제거
    text = text.replace("\n", " ")

    # 대괄호 포함 텍스트 제거: [파이낸셜뉴스], [사진] 등
    text = re.sub(r"\[[^\]]*\]", "", text)

    # '/사진', '/사진제공' 제거
    text = re.sub(r"/사진(제공)?", "", text)

    # 이메일 주소 제거 (예: josh@yna.co.kr)
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "", text)

    # 문장 단위 분리 (간단하게 마침표 기준, 필요시 KSS 등 적용 가능)
    sentences = split_sentences(text)

    # 제거할 패턴들 (뉴스 문장에서 자주 등장하는 패턴)
    patterns = [
        r"(자세한 내용|자세한 사항)",  # 뉴스 기본 표현
        r"\d{4}[.-]\d{1,2}[.-]\d{1,2}",  # 날짜 (예: 2025.03.26, 2024-12-01)
        r"([0-9,]+(?:만)?[0-9,]*\s?(?:원|만원))",  # 가격 (예: 3,500원, 12000원)
        r"(강세|펀드|시가총액|등락률|한국거래소)",  # 증시 용어
        r"\([+-]?[0-9.,]+%\)",  # 괄호 안 퍼센트 등락률
        r"(투자의견|연구원|평가|예상치|증권가|리포트|팀장)",  # 애널리스트 용어
        r"(순이익|전년|매출|영업이익|영업적자|증시|코스피|코스닥|다우|나스닥|매출액|거래일|호조세|레버리지|투자자|조정|자산|수익률|이익률|수익성|내리막|부진한|낙폭|기대치|실적발표|기업 가치)",  # 시장 용어
    ]

    # 하나의 통합 패턴으로 컴파일
    combined_pattern = re.compile("|".join(patterns))

    # 필터링된 문장만 유지
    filtered = [s for s in sentences if not combined_pattern.search(s)]

    text_preprocessed = " ".join(filtered)

    # print(f"원문:{sentences}\n|\n전처리 된 문장: {text_preprocessed}\n\n")

    return text_preprocessed


def summarize_event_focused(
    text,
    encoder_sess,
    decoder_sess,
    tokenizer,
    max_length=128,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
):
    input_ids = tokenizer.encode(text).ids
    input_ids_np = np.array([input_ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids_np, dtype=np.int64)

    encoder_outputs = encoder_sess.run(
        None, {"input_ids": input_ids_np, "attention_mask": attention_mask}
    )[0]

    decoder_input_ids = [tokenizer.token_to_id("<s>")]
    generated_ids = decoder_input_ids.copy()

    for _ in range(max_length):
        decoder_input_np = np.array([generated_ids], dtype=np.int64)
        decoder_inputs = {
            "input_ids": decoder_input_np,
            "encoder_hidden_states": encoder_outputs,
            "encoder_attention_mask": attention_mask,
        }
        logits = decoder_sess.run(None, decoder_inputs)[0]
        next_token_logits = logits[:, -1, :]

        # repetition penalty 적용
        for token_id in set(generated_ids):
            next_token_logits[0, token_id] /= repetition_penalty

        # no_repeat_ngram_size 적용
        if no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size:
            ngram = tuple(generated_ids[-(no_repeat_ngram_size - 1) :])
            banned = {
                tuple(generated_ids[i : i + no_repeat_ngram_size])
                for i in range(len(generated_ids) - no_repeat_ngram_size + 1)
            }
            for token_id in range(next_token_logits.shape[-1]):
                if ngram + (token_id,) in banned:
                    next_token_logits[0, token_id] = -1e9  # 큰 마이너스

        # greedy 선택
        next_token_id = int(np.argmax(next_token_logits, axis=-1)[0])

        if next_token_id == tokenizer.token_to_id("</s>"):
            break

        generated_ids.append(next_token_id)

    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return summary


def get_ner_tokens(tokenizer, session, text, id2label):
    # 🟡 토큰화 및 입력값 준비
    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    # 🔵 ONNX 추론 실행
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    logits = session.run(None, inputs)[0]  # shape: (1, seq_len, num_labels)

    # 🔵 라벨 인덱스 → 실제 라벨명
    preds = np.argmax(logits, axis=-1)[0]
    labels = [id2label[p] for p in preds[: len(encoding.tokens)]]

    # 🔵 시각화
    tokens = encoding.tokens

    return tokens, labels


def extract_ogg_economy(tokens, labels, target_label="OGG_ECONOMY"):
    merged_words = []
    current_word = ""

    for token, label in zip(tokens, labels):
        token_clean = token.replace("##", "") if token.startswith("##") else token

        if label == f"B-{target_label}":
            if current_word:
                merged_words.append(current_word)
            current_word = token_clean

        elif label == f"I-{target_label}":
            current_word += token_clean

        else:
            if current_word:
                merged_words.append(current_word)
                current_word = ""

    if current_word:
        merged_words.append(current_word)

    stock_list = merged_words.copy()

    return stock_list


# 종목명 집합 불러오기
def load_official_stock_list(krx_csv_path):
    df = pd.read_csv(krx_csv_path, encoding="cp949")
    return list(set(df["종목명"].dropna().unique()))


# 종목 리스트에서 공식 종목만 필터링
def filter_official_stocks_from_list(stock_list, official_stock_set):
    return [stock for stock in stock_list if stock in official_stock_set]


# 종목 → 업종 매핑 딕셔너리 생성
def load_stock_to_industry_map(kospi_desc_csv_path):
    df = pd.read_csv(kospi_desc_csv_path, encoding="cp949")
    return dict(zip(df["종목명"], df["업종명"]))


# 종목 리스트를 업종 리스트로 변환
def get_industry_list_from_stocks(stock_list, stock_to_industry):
    if len(stock_list) > 4 or len(stock_list) < 1:
        return []

    return [
        stock_to_industry.get(stock, "")
        for stock in stock_list
        if stock_to_industry.get(stock, "") != ""
    ]


def get_news_deduplicate_by_title(news_list):
    seen_titles = set()
    deduped_news = []

    for news in news_list:
        title = news.get("title")
        if title not in seen_titles:
            seen_titles.add(title)
            deduped_news.append(news)

    return deduped_news


def predict_topic_for_df(df, vectorizer, lda_model, stopwords, n_topics=9):
    # 2. 형태소 분석기 및 불용어 로드
    okt = Okt()

    # 3. 텍스트 정제 함수
    def clean_text(text):
        text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
        text = re.sub(r"[^가-힣\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # 4. 명사 추출 함수
    def extract_nouns(text):
        nouns = okt.nouns(text)
        nouns = [word for word in nouns if word not in stopwords and len(word) > 1]
        return " ".join(nouns)

    # 5. 데이터 전처리 (정제 + 명사 추출)
    processed_texts = [extract_nouns(clean_text(text)) for text in df["summary"]]

    # 6. 벡터라이즈 (DTM 생성)
    new_dtm = vectorizer.transform(processed_texts)

    # 7. LDA 토픽 분포 예측
    topic_distribution = lda_model.transform(new_dtm)

    # 8. 결과 DataFrame 생성 (news_id, 주요 토픽, 토픽1~토픽n)
    topic_columns = [f"토픽 {i+1}" for i in range(n_topics)]
    topic_data = np.concatenate([topic_distribution], axis=1)
    topic_df = pd.DataFrame(topic_data, columns=topic_columns)
    topic_df["news_id"] = df["news_id"].values

    # 9. news_id 기준으로 merge
    result_df = pd.merge(df, topic_df, on="news_id", how="left")

    return result_df


def send_to_redis(news_data):
    try:
        r = redis.Redis(
            host="43.200.17.139",
            port=6379,
            password="q1w2e3r4!@#",
            decode_responses=True,  # bytes 대신 str 로 받기
        )
        if not r.ping():
            log.error("Redis 서버에 연결할 수 없습니다.")
            return

        # Redis에 저장
        for news in news_data:
            channel = "news-channel"
            data = {
                "news_id": news["news_id"],
                "wdate": news["wdate"],
                "title": news["title"],
                "article": news["article"],
                "press": news["press"],
                "url": news["url"],
                "image": news["image"],
            }
            message = json.dumps(data, ensure_ascii=False)
            r.publish(channel, message)

        log.info(f"Redis에 {len(news_data)}건 뉴스 푸시 완료")
    except Exception as e:
        log.error(f"Redis 푸시 실패 ({type(e).__name__}): {e}")


if __name__ == "__main__":
    log.info("로그 테스트: news_pipeline.py 직접 실행됨")
    fetch_latest_news()
