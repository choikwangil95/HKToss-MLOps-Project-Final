from news_pipeline import fetch_latest_news
import schedule
import time
import logging

log = logging.getLogger("news_logger")

def job():
    log.info("🕒 [스케줄러] 뉴스 수집 실행")

    # ──────────────────────────────
    # 1 뉴스 실시간 수집
    # - 1분마다 수집
    # ──────────────────────────────
    
    # 1 뉴스 실시간 수집 실행 함수
    news_crawled = fetch_latest_news()


    # ──────────────────────────────
    # 2 뉴스 전처리
    # - 본문 전처리 및 요약, 종목과 업종명 매칭
    # ──────────────────────────────
    
    # 1 뉴스 본문 전처리 실행 함수

    # 2 뉴스 본문 요악 함수

    # 3 뉴스 종목 매칭 함수

    # 4 뉴스 업종면 매칭 함수

    # ──────────────────────────────
    # 3 뉴스 경제 및 행동 지표 피쳐 추가
    # - 주가 D+1~D+30 변동률, 금리, 환율, 기관 매매동향, 유가 등
    # ──────────────────────────────

    # 으악

    # ──────────────────────────────
    # 4 뉴스 시멘틱 피쳐 추가
    # - topic별 분포값, 클러스터 동일 여부
    # ──────────────────────────────

    # 으악

if __name__ == "__main__":
    log.info("✅ run_scheduler.py 시작됨")

    # 🟡 첫 실행 즉시
    job()

    # 🟢 이후 매 1분마다 실행
    schedule.every(1).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
