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
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from label_map import label2id, id2label
import pandas as pd
import re
from konlpy.tag import Okt
import numpy as np

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
formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

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
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
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
        image = image_tag["content"] if image_tag and image_tag.has_attr("content") else None

        article_tag = soup.select_one("article#dic_area")
        article = article_tag.get_text(strip=True, separator="\n") if article_tag else ""

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
        DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/news_db")
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
        DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/news_db")
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        insert_query = """
		INSERT INTO news_v2_metadata (news_id, stock_list, industry_list)
		VALUES (%s, %s, %s)
		ON CONFLICT (news_id) DO NOTHING;
		"""

        values = [
            (
                article["news_id"],
                article["stock_list"],
                article["industry_list"],
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
                log.info(f"[NEW] {article['wdate']} - {article['title']} ({article['press']})")
                log.info(f"{preview}...\n")
            except Exception as e:
                log.error(f"본문 파싱 실패 ({type(e).__name__}): {e}")
    else:
        log.info("새 뉴스 없음")

    if new_articles_crawled:
        save_to_db(new_articles_crawled)

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


def get_summarize_model(model_name="digit82/kobart-summarization", model_dir="./models/kobart_summary"):
    # 모델 이름 & 로컬 저장 경로
    model_name = "digit82/kobart-summarization"
    model_dir = "./models/kobart_summary"

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 로컬에 저장된 모델이 있으면 불러오고, 없으면 다운로드 후 저장
    if os.path.exists(model_dir):
        print("📦 로컬 모델 로드 중...")
        tokenizer_summarize = AutoTokenizer.from_pretrained(model_dir)
        model_summarize = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        print("🌐 모델 다운로드 중...")
        tokenizer_summarize = AutoTokenizer.from_pretrained(model_name)
        model_summarize = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)

        print("💾 모델 로컬 저장...")
        tokenizer_summarize.save_pretrained(model_dir)
        model_summarize.save_pretrained(model_dir)

    # 모델 디바이스 할당
    model_summarize.to(device)

    return tokenizer_summarize, model_summarize, device


def summarize_event_focused(text, tokenizer_summarize, model_summarize, device):
    # 토크나이징 및 텐서 변환 (GPU로 올리기)
    inputs = tokenizer_summarize(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    text_length = len(text)

    # 1. 최소 길이는 짧은 본문은 고정, 긴 본문은 점진적으로 증가
    def compute_min_length(text_length: int) -> int:
        if text_length < 300:
            return 30
        else:
            return 50

    min_len = compute_min_length(text_length)
    max_len = round(text_length * 0.5) + 50  # 더 여유를 주되 max 길이 제한

    # 2. generate 최적 설정
    summary_ids = model_summarize.generate(
        input_ids,
        min_length=min_len,
        max_new_tokens=max_len,
        num_beams=4,  # 4보다 빠름. 품질도 비슷
        length_penalty=1.0,  # 길이 패널티 완화
        repetition_penalty=1.3,  # 반복 억제 강화
        no_repeat_ngram_size=3,  # 반복 문장 방지
        early_stopping=True,
        do_sample=False,  # 일관된 요약
    )

    article_summarized = tokenizer_summarize.decode(summary_ids[0], skip_special_tokens=True)

    return article_summarized


def get_ner_pipeline(model_name="KPF/KPF-BERT-NER", local_dir="models/ner"):
    # 디바이스 설정
    device = 0 if torch.cuda.is_available() else -1

    # 로컬 경로에 모델이 없으면 다운로드
    if not os.path.exists(local_dir):
        print("🔽 NER 모델 로컬에 없음. 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, use_safetensors=True)

        # 저장
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
    else:
        print("📦 로컬에서 NER 모델 불러오는 중...")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForTokenClassification.from_pretrained(local_dir)

    # 라벨 매핑
    model.config.label2id = label2id
    model.config.id2label = id2label

    # max length 설정
    tokenizer.model_max_length = 512

    # NER pipeline 생성
    ner_pipe = pipeline(
        task="ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        framework="pt",
        device=device,
    )

    return ner_pipe


def extract_ner(ner_pipeline, text):
    entities = ner_pipeline(text)
    results = []
    seen = set()

    for ent in entities:
        word = ent["word"].replace("##", "").strip()
        tag = ent["entity_group"]

        score = ent["score"]

        if word and score >= 0.95 and (word, tag) not in seen:
            results.append((word, tag))
            seen.add((word, tag))

    return results


def get_stock_names(ner_pipeline, text):
    ner_list = extract_ner(ner_pipeline, text)

    # OGG_ECONOMY만 필터링하여 종목명만 리스트로 추출
    stock_names = [ent[0] for ent in ner_list if ent[1] == "OGG_ECONOMY"]

    return stock_names


# 종목명 집합 불러오기
def load_official_stock_list(krx_csv_path):
    df = pd.read_csv(krx_csv_path, encoding="cp949")
    return set(df["종목명"].dropna().unique())


# 종목 리스트에서 공식 종목만 필터링
def filter_official_stocks_from_list(stock_list, official_stock_set):
    return [stock for stock in stock_list if stock in official_stock_set]


# 종목 → 업종 매핑 딕셔너리 생성
def load_stock_to_industry_map(kospi_desc_csv_path):
    df = pd.read_csv(kospi_desc_csv_path, encoding="cp949")
    return dict(zip(df["종목명"], df["업종명"]))


# 종목 리스트를 업종 리스트로 변환
def get_industry_list_from_stocks(stock_list, stock_to_industry):
    return [stock_to_industry.get(stock, "") for stock in stock_list if stock_to_industry.get(stock, "") != ""]


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


if __name__ == "__main__":
    log.info("로그 테스트: news_pipeline.py 직접 실행됨")
    fetch_latest_news()
