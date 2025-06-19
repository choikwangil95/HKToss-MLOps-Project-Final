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
import numpy as np
import json
import redis
from dotenv import load_dotenv
import ast
from pykrx import stock
from datetime import timedelta

load_dotenv()

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

    conn = None  # ✅ 먼저 None으로 초기화
    cur = None

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
		INSERT INTO news_v2_metadata (news_id, summary, stock_list, stock_list_view, industry_list, impact_score)
		VALUES (%s, %s, %s, %s, %s, %s)
		ON CONFLICT (news_id) DO NOTHING;
		"""

        values = [
            (
                article["news_id"],
                article["summary"],
                (
                    json.dumps(article["stock_list"], ensure_ascii=False)
                    if article["stock_list"] is not None
                    else None
                ),
                (
                    json.dumps(article["stock_list_view"], ensure_ascii=False)
                    if article["stock_list_view"] is not None
                    else None
                ),
                (
                    json.dumps(article["industry_list"], ensure_ascii=False)
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


def save_to_db_topics(articles):
    if not articles:
        log.info("저장할 뉴스 없음")
        return

    insert_query = """
    INSERT INTO news_v2_topic (news_id, topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7, topic_8, topic_9)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (news_id) DO NOTHING;
    """

    values = [
        (
            article["news_id"],
            article["topic_1"],
            article["topic_2"],
            article["topic_3"],
            article["topic_4"],
            article["topic_5"],
            article["topic_6"],
            article["topic_7"],
            article["topic_8"],
            article["topic_9"],
        )
        for article in articles
    ]

    try:
        DB_URL = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/news_db"
        )
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        execute_batch(cur, insert_query, values)
        conn.commit()

        log.info(f"🧾 Topic DB 저장 완료: {len(values)}건 저장")

    except Exception as e:
        log.error(f"❌ Topic DB 저장 중 오류 ({type(e).__name__}): {e}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def save_to_db_external(market_datas):
    if not market_datas:
        log.info("저장할 뉴스 없음")
        return

    columns = [
        "news_id",
        "d_minus_5_date_close",
        "d_minus_5_date_volume",
        "d_minus_5_date_foreign",
        "d_minus_5_date_institution",
        "d_minus_5_date_individual",
        "d_minus_4_date_close",
        "d_minus_4_date_volume",
        "d_minus_4_date_foreign",
        "d_minus_4_date_institution",
        "d_minus_4_date_individual",
        "d_minus_3_date_close",
        "d_minus_3_date_volume",
        "d_minus_3_date_foreign",
        "d_minus_3_date_institution",
        "d_minus_3_date_individual",
        "d_minus_2_date_close",
        "d_minus_2_date_volume",
        "d_minus_2_date_foreign",
        "d_minus_2_date_institution",
        "d_minus_2_date_individual",
        "d_minus_1_date_close",
        "d_minus_1_date_volume",
        "d_minus_1_date_foreign",
        "d_minus_1_date_institution",
        "d_minus_1_date_individual",
        "d_plus_1_date_close",
        "d_plus_2_date_close",
        "d_plus_3_date_close",
        "d_plus_4_date_close",
        "d_plus_5_date_close",
        "fx",
        "bond10y",
        "base_rate",
    ]

    insert_query = f"""
        INSERT INTO news_v2_external ({', '.join(columns)})
        VALUES ({', '.join(['%s'] * len(columns))})
        ON CONFLICT (news_id) DO NOTHING;
    """

    values = [
        tuple(market_data.get(col, None) for col in columns)
        for market_data in market_datas
    ]

    try:
        DB_URL = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/news_db"
        )
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        execute_batch(cur, insert_query, values)
        conn.commit()

        log.info(f"🧾 External DB 저장 완료: {len(values)}건 저장")

    except Exception as e:
        log.error(f"❌ External DB 저장 중 오류 ({type(e).__name__}): {e}")

    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass


def update_db_impact_score(score_datas):
    if not score_datas:
        log.info("업데이트할 데이터 없음")
        return

    update_query = """
        UPDATE news_v2_metadata
        SET impact_score = %s,
            d_plus_1_date_close = %s,
            d_plus_2_date_close = %s,
            d_plus_3_date_close = %s,
            d_plus_4_date_close = %s,
            d_plus_5_date_close = %s
        WHERE news_id = %s;
    """

    values = [
        (
            data["score"],
            data["d_plus"][0],
            data["d_plus"][1],
            data["d_plus"][2],
            data["d_plus"][3],
            data["d_plus"][4],
            data["news_id"],
        )
        for data in score_datas
    ]

    try:
        DB_URL = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/news_db"
        )
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        execute_batch(cur, update_query, values)
        conn.commit()

        log.info(f"🧾 Impact Score 업데이트 완료: {len(values)}건")

    except Exception as e:
        log.error(f"❌ Impact Score 업데이트 오류 ({type(e).__name__}): {e}")

    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass


def update_db_external(score_datas):
    if not score_datas:
        log.info("업데이트할 데이터 없음")
        return

    update_query = """
        UPDATE news_v2_external
        SET d_plus_1_date_close = %s,
            d_plus_2_date_close = %s,
            d_plus_3_date_close = %s,
            d_plus_4_date_close = %s,
            d_plus_5_date_close = %s
        WHERE news_id = %s;
    """

    values = [
        (
            data["d_plus"][0],
            data["d_plus"][1],
            data["d_plus"][2],
            data["d_plus"][3],
            data["d_plus"][4],
            data["news_id"],
        )
        for data in score_datas
    ]

    try:
        DB_URL = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/news_db"
        )
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        execute_batch(cur, update_query, values)
        conn.commit()

        log.info(f"🧾 External 업데이트 완료: {len(values)}건")

    except Exception as e:
        log.error(f"❌ External 업데이트 오류 ({type(e).__name__}): {e}")

    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass


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
            except Exception as e:
                log.error(f"본문 파싱 실패 ({type(e).__name__}): {e}")
    else:
        log.info("새 뉴스 없음")

    return new_articles_crawled


def enrich_stock_list(stock_names_raw, stock_name_to_code):
    try:
        stock_names = stock_names_raw
        result = []
        for name in stock_names:
            code = stock_name_to_code.get(name)
            if code:
                result.append({"stock_id": str(code), "stock_name": name})
        return result
    except Exception:
        return []


def extract_industries(stock_list, code_to_industry):
    if len(stock_list) == 0:
        return []

    industries = []
    seen = set()
    for stock in stock_list:
        stock_id = stock.get("stock_id")
        if stock_id is None:
            continue
        industry = code_to_industry.get(int(stock_id))
        if industry:
            key = (stock_id, industry["industry_id"])
            if key not in seen:
                seen.add(key)
                industries.append(
                    {
                        "stock_id": str(stock_id),
                        "industry_id": industry["industry_id"],
                        "industry_name": industry["industry_name"],
                    }
                )
    return industries


# ──────────────────────────────
# 📌 뉴스 수집 메인 함수
# ──────────────────────────────


def remove_market_related_sentences(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 줄바꿈 제거
    text = text.replace("\n", " ")

    # 대괄호 포함 텍스트 제거: [파이낸셜뉴스], [사진] 등
    text = re.sub(r"\[[^\]]*\]", "", text)

    # '/사진', '/사진제공' 제거
    text = re.sub(r"/사진(제공)?", "", text)

    # 이메일 주소 제거 (예: josh@yna.co.kr)
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "", text)

    # 문장 단위 분리 (간단하게 마침표 기준, 필요시 KSS 등 적용 가능)
    try:
        sentences = split_sentences(text)
    except Exception as e:
        print(f"[❗KSS 오류] 문장 분리 실패: {e}")
        sentences = text.split(". ")

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


def get_article_summary(
    text,
):
    url = "http://15.165.211.100:8000/plm/summarize"  # 또는 EC2 내부/외부 주소
    payload = {"article": text}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        summary = response.json()["summary"]  # 또는 실제 리턴 필드에 따라 조정

        return summary

    except Exception as e:
        print(f"❌ 요약 요청 실패: {e}")
        return ""


def get_stock_list(text):
    # 🟡 토큰화 및 입력값 준비

    url = "http://15.165.211.100:8000/plm/stocks"
    payload = {"article": text}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["stock_list"]  # 혹은 API 응답 구조에 따라 조정
    except Exception as e:
        print(f"❌ 종목명 추출 실패: {e}")
        return []


def get_lda_topic(text):
    # 🟡 토큰화 및 입력값 준비

    url = "http://15.165.211.100:8000/news/topics"
    payload = {"article": text}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["lda_topics"]  # 혹은 API 응답 구조에 따라 조정
    except Exception as e:
        print(f"❌ 종목명 추출 실패: {e}")
        return []


# 종목명 집합 불러오기
def load_official_stock_list(krx_csv_path):
    df = pd.read_csv(krx_csv_path, encoding="cp949")

    stock_list = list(set(df["종목명"].dropna().unique()))
    stock_name_to_code = dict(zip(df["종목명"], df["종목코드"]))
    return stock_list, stock_name_to_code


# 종목 리스트에서 공식 종목만 필터링
def filter_official_stocks_from_list(stock_list, official_stock_set):
    return [stock for stock in stock_list if stock in official_stock_set]


# 종목 → 업종 매핑 딕셔너리 생성
def load_stock_to_industry_map(kospi_desc_csv_path):
    df = pd.read_csv(kospi_desc_csv_path, encoding="cp949")

    industry_list = dict(zip(df["종목명"], df["업종명"]))

    code_to_industry = {
        row["종목코드"]: {
            "industry_id": str(row["업종코드"]),
            "industry_name": row["업종명"],
        }
        for _, row in df.iterrows()
    }

    return industry_list, code_to_industry


# 기준금리 데이터프레임 로드
def load_rate_df(rate_path):
    rate_df = pd.read_csv(rate_path)
    rate_df["date"] = pd.to_datetime(rate_df["date"])
    rate_df = rate_df.sort_values("date")

    return rate_df


# 종목 리스트를 업종 리스트로 변환
def get_industry_list_from_stocks(stock_list, stock_to_industry):
    # 조건: 종목 수가 1~4개가 아닐 경우 빈 리스트 반환
    if not (1 <= len(stock_list) <= 4):
        return []

    return [
        stock_to_industry.get(stock["stock_id"], "")
        for stock in stock_list
        if stock_to_industry.get(stock["stock_id"], "") != ""
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

            wdate = datetime.strptime(news["wdate"], "%Y-%m-%d %H:%M").isoformat()
            data = {
                "news_id": news["news_id"],
                "wdate": wdate,
                "title": news["title"],
                "article": news["article"],
                "press": news["press"],
                "url": news["url"],
                "image": news["image"],
                "impact_score": news["impact_score"],
            }
            message = json.dumps(data, ensure_ascii=False)
            r.publish(channel, message)

        log.info(f"Redis에 {len(news_data)}건 뉴스 푸시 완료")
    except Exception as e:
        log.error(f"Redis 푸시 실패 ({type(e).__name__}): {e}")


class NewsMarketPipeline:

    def __init__(self, news_list, df_base_rate):
        self.api_key = os.getenv("KOREA_BANK_API_KEY")

        self.df = pd.DataFrame(news_list)
        self.ticker_name_map = None
        self.trading_days = None
        self.ohlcv_dict = {}
        self.trading_dict = {}
        self.fx_df = None
        self.bond_df = None
        self.rate_df = df_base_rate

    def get_df(self):
        return self.df

    def extract_stock_name(self):
        if "stock_list" not in self.df.columns:
            raise Exception(
                "stock_list 컬럼이 없습니다. 실제 컬럼: "
                + str(self.df.columns.tolist())
            )

        def get_last_stock_name(x):
            try:
                items = ast.literal_eval(x) if isinstance(x, str) else x
                return items[-1]["stock_name"] if items else None
            except:
                return None

        self.df["stock_name"] = self.df["stock_list"].apply(get_last_stock_name)

    def add_news_date(self):
        if "wdate" in self.df.columns:
            self.df["wdate"] = pd.to_datetime(self.df["wdate"])
            self.df["news_date"] = self.df["wdate"].dt.normalize()
        elif "news_date" in self.df.columns:
            self.df["news_date"] = pd.to_datetime(self.df["news_date"])
        else:
            raise Exception(
                "wdate/news_date 컬럼이 없습니다. 실제 컬럼: "
                + str(self.df.columns.tolist())
            )

    def get_ticker_name_map(self, recent_date="2025-05-30"):
        kospi_tickers = stock.get_market_ticker_list(date=recent_date, market="KOSPI")
        return {
            stock.get_market_ticker_name(ticker): ticker for ticker in kospi_tickers
        }

    def add_ticker(self):
        if self.ticker_name_map is None:
            self.ticker_name_map = self.get_ticker_name_map()

        self.df["ticker"] = self.df["stock_name"].apply(
            lambda name: self.ticker_name_map.get(name) if pd.notna(name) else None
        )

    def get_trading_days(self, start_year=2022, end_year=2026):
        days = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                try:
                    days_this_month = stock.get_previous_business_days(year=y, month=m)
                    days.extend(days_this_month)
                except:
                    pass
        return pd.to_datetime(sorted(set(days)))

    def adjust_to_nearest_trading_day(self, date):
        idx = self.trading_days.searchsorted(date, side="right") - 1
        if idx >= 0:
            return self.trading_days[idx]
        return pd.NaT

    def add_trading_dates(self):
        if self.trading_days is None:
            self.trading_days = self.get_trading_days()

        self.df["d_day_date"] = self.df["news_date"].apply(
            self.adjust_to_nearest_trading_day
        )

        offsets = {
            "d_minus_5_date": -5,
            "d_minus_4_date": -4,
            "d_minus_3_date": -3,
            "d_minus_2_date": -2,
            "d_minus_1_date": -1,
            "d_day_date": 0,
        }

        def fill_offsets(row):
            d_day = row["d_day_date"]
            if not pd.isna(d_day):
                weekday = d_day.weekday()
                if weekday == 5:
                    d_day = self.adjust_to_nearest_trading_day(
                        d_day - timedelta(days=1)
                    )
                elif weekday == 6:
                    d_day = self.adjust_to_nearest_trading_day(
                        d_day - timedelta(days=2)
                    )

            res = {}
            if pd.isna(d_day):
                for k in offsets:
                    res[k] = pd.NaT
                return pd.Series(res)

            idx = self.trading_days.searchsorted(d_day)
            for k, v in offsets.items():
                i = idx + v
                res[k] = (
                    self.trading_days[i] if 0 <= i < len(self.trading_days) else pd.NaT
                )
            return pd.Series(res)

        df_offsets = self.df.apply(fill_offsets, axis=1)
        self.df = pd.concat(
            [self.df.reset_index(drop=True), df_offsets.reset_index(drop=True)], axis=1
        )

    def fetch_ohlcv_and_trading(self):
        offsets = [f"d_minus_{i}_date" for i in range(1, 6)]
        all_dates = (
            pd.concat([self.df[col] for col in offsets], ignore_index=True)
            .dropna()
            .unique()
        )
        all_dates_str = sorted(
            [pd.to_datetime(d).strftime("%Y%m%d") for d in all_dates]
        )
        tickers = self.df["ticker"].dropna().unique().tolist()

        for ticker in tickers:
            try:
                self.ohlcv_dict[ticker] = stock.get_market_ohlcv_by_date(
                    min(all_dates_str), max(all_dates_str), ticker
                )
            except:
                pass
            try:
                self.trading_dict[ticker] = stock.get_market_trading_value_by_date(
                    min(all_dates_str), max(all_dates_str), ticker
                )
            except:
                pass

    def add_ohlcv_and_trading(self):
        offsets = [f"d_minus_{i}_date" for i in range(1, 6)]

        all_ohlcv_rows = []
        for ticker, df in self.ohlcv_dict.items():
            df = df.reset_index().rename(columns={"날짜": "date"})
            df["ticker"] = ticker
            all_ohlcv_rows.append(df[["date", "ticker", "종가", "거래량"]])
        df_ohlcv_all = pd.concat(all_ohlcv_rows) if all_ohlcv_rows else pd.DataFrame()

        all_trading_rows = []
        for ticker, df in self.trading_dict.items():
            df = df.reset_index().rename(columns={"날짜": "date"})
            df["ticker"] = ticker
            df = df[["date", "ticker", "외국인합계", "기관합계", "개인"]]
            all_trading_rows.append(df)
        df_trading_all = (
            pd.concat(all_trading_rows) if all_trading_rows else pd.DataFrame()
        )

        for col in offsets:
            self.df = (
                self.df.merge(
                    df_ohlcv_all,
                    how="left",
                    left_on=[col, "ticker"],
                    right_on=["date", "ticker"],
                )
                .rename(columns={"종가": f"{col}_close", "거래량": f"{col}_volume"})
                .drop(columns="date")
            )
            self.df = (
                self.df.merge(
                    df_trading_all,
                    how="left",
                    left_on=[col, "ticker"],
                    right_on=["date", "ticker"],
                )
                .rename(
                    columns={
                        "외국인합계": f"{col}_foreign",
                        "기관합계": f"{col}_institution",
                        "개인": f"{col}_individual",
                    }
                )
                .drop(columns="date")
            )

    def fetch_fx(self, start_date, end_date):
        if self.fx_df is not None:
            return self.fx_df
        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{self.api_key}/json/kr/1/1000/731Y001/D/{start_date}/{end_date}/0000001/"
        resp = requests.get(url).json()
        if "StatisticSearch" not in resp or "row" not in resp["StatisticSearch"]:
            return pd.DataFrame()
        df = pd.DataFrame(resp["StatisticSearch"]["row"])
        df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d")
        df["usdkrw"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
        self.fx_df = df[["date", "usdkrw"]].sort_values("date")
        return self.fx_df

    def fetch_bond10y(self, start_date, end_date):
        if self.bond_df is not None:
            return self.bond_df
        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{self.api_key}/json/kr/1/1000/817Y002/D/{start_date}/{end_date}/010200000/"
        resp = requests.get(url).json()
        if "StatisticSearch" not in resp or "row" not in resp["StatisticSearch"]:
            return pd.DataFrame()
        df = pd.DataFrame(resp["StatisticSearch"]["row"])
        df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d")
        df["bond10y"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
        self.bond_df = df[["date", "bond10y"]].sort_values("date")
        return self.bond_df

    def add_external_vars(self):
        self.df = self.df.sort_values("news_date")
        if self.trading_days is None:
            self.trading_days = self.get_trading_days()
        raw_start = self.df["news_date"].min() - timedelta(days=1)
        raw_end = self.df["news_date"].max() - timedelta(days=1)
        start_date = self.adjust_to_nearest_trading_day(raw_start)
        end_date = self.adjust_to_nearest_trading_day(raw_end)
        if pd.isna(start_date) or pd.isna(end_date):
            return

        start_str, end_str = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        fx_df = self.fetch_fx(start_str, end_str)
        bond_df = self.fetch_bond10y(start_str, end_str)

        if not fx_df.empty:
            self.df = pd.merge_asof(
                self.df,
                fx_df.rename(columns={"date": "news_date", "usdkrw": "fx"}),
                on="news_date",
                direction="backward",
            )
        if not bond_df.empty:
            self.df = pd.merge_asof(
                self.df,
                bond_df.rename(columns={"date": "news_date"}),
                on="news_date",
                direction="backward",
            )
        if self.rate_df is not None and not self.rate_df.empty:
            self.df = pd.merge_asof(
                self.df,
                self.rate_df.rename(columns={"date": "news_date", "rate": "base_rate"}),
                on="news_date",
                direction="backward",
            )

    def run(self):
        steps = [
            ("extract_stock_name", self.extract_stock_name),
            ("add_news_date", self.add_news_date),
            ("add_ticker", self.add_ticker),
            ("add_trading_dates", self.add_trading_dates),
            ("fetch_ohlcv_and_trading", self.fetch_ohlcv_and_trading),
            ("add_ohlcv_and_trading", self.add_ohlcv_and_trading),
            ("add_external_vars", self.add_external_vars),
        ]

        for step_name, func in steps:
            try:
                func()
            except Exception as e:
                print(f"[ERROR] Step '{step_name}' failed: {e}")

        try:
            self.df = self.df.drop(
                columns=["wdate", "stock_list", "stock_name", "news_date", "ticker"]
                + [f"d_minus_{i}_date" for i in range(1, 6)]
                + ["d_day_date"],
                errors="ignore",
            )
        except Exception as e:
            print(f"[WARN] Drop columns failed: {e}")

        try:
            # 기준 컬럼들
            prefixes = ["close", "volume", "institution", "foreign", "individual"]
            days = ["d_minus_5", "d_minus_4", "d_minus_3", "d_minus_2"]

            # 등락률 계산
            for prefix in prefixes:
                d_minus_1_col = f"d_minus_1_date_{prefix}"

                # d-1 컬럼에 NaN 있으면 전체 스킵
                if self.df[d_minus_1_col].isna().any():
                    continue

                for day in days:
                    col = f"{day}_date_{prefix}"

                    # 비교 대상 컬럼에 NaN 있으면 해당 day 루프 스킵
                    if self.df[col].isna().any():
                        continue

                    self.df[col] = np.round(
                        (self.df[d_minus_1_col] - self.df[col])
                        / self.df[d_minus_1_col],
                        2,
                    )

            self.df["d_minus_1_date_close"] = 0
            self.df["d_minus_1_date_volume"] = 0
            self.df["d_minus_1_date_institution"] = 0
            self.df["d_minus_1_date_foreign"] = 0
            self.df["d_minus_1_date_individual"] = 0

            return self.df.to_dict(orient="records")
        except Exception as e:
            print(f"[ERROR] Converting to dict failed: {e}")
            return []


def request_impact_score(news_id):
    try:
        url = f"http://15.165.211.100:8000/news/{news_id}/impact_score"

        r = requests.get(url, timeout=5)
        r.raise_for_status()

        d_plus = r.json().get("d_plus", [])
        impact_score = r.json().get("impact_score", 0)

        return {"news_id": news_id, "score": impact_score, "d_plus": d_plus}

    except Exception as e:
        print(f"❌ {news_id} 실패: {e}")

        return {"news_id": news_id, "score": 0, "d_plus": []}


def get_impact_score(market_datas):
    return [request_impact_score(md["news_id"]) for md in market_datas]


if __name__ == "__main__":
    log.info("로그 테스트: news_pipeline.py 직접 실행됨")
    fetch_latest_news()
