# news_pipeline.py
import requests, random, time, os, logging, concurrent.futures
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from requests.adapters import HTTPAdapter, Retry
from logging.handlers import RotatingFileHandler
import psycopg2
from psycopg2.extras import execute_batch

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
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 🔧 파일 핸들러 설정
log_path = os.path.join(LOG_DIR, "news.log")
file_handler = RotatingFileHandler(
    log_path,
    maxBytes=5 * 1024 * 1024,  # 5MB 넘으면 순환
    backupCount=3             # 최대 3개 백업 보관
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

NEWS_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402"
LAST_CRAWLED_FILE = "automation/data/last_crawled.txt"

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
                article["image"]
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

# ──────────────────────────────
# 📌 뉴스 수집 메인 함수
# ──────────────────────────────

def fetch_latest_news():
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
                new_articles.append({
                    "wdate": wdate,
                    "title": title,
                    "press": press,
                    "url": url,
                })
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

                wdate = article['wdate']
                title = article['title']
                press = article['press']
                url = article['url']

                article_time_dt = parse_wdate(wdate)
                date_str = article_time_dt.strftime("%Y%m%d")

                news_id = generate_news_id(date_str)

                new_articles_crawled.append({
                    "news_id": news_id,
                    "wdate": wdate,
                    "title": title,
                    "article": article_text,
                    "press": press,
                    "url": url,
                    "image": image,
                })

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

if __name__ == "__main__":
    log.info("로그 테스트: news_pipeline.py 직접 실행됨")
    fetch_latest_news()