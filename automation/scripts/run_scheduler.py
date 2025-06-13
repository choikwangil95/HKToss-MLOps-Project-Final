from news_pipeline import (
    NewsMarketPipeline,
    fetch_latest_news,
    get_article_summary,
    get_stock_list,
    load_rate_df,
    remove_market_related_sentences,
    load_official_stock_list,
    filter_official_stocks_from_list,
    load_stock_to_industry_map,
    get_industry_list_from_stocks,
    save_to_db_external,
    save_to_db_metadata,
    get_news_deduplicate_by_title,
    save_to_db,
    send_to_redis,
)
import schedule
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("news_logger")


def job(official_stock_set, stock_to_industry, df_base_rate):
    log.info("🕒 [스케줄러] 뉴스 수집 실행")

    # ──────────────────────────────
    # 1 뉴스 실시간 수집
    # - 1분마다 수집
    # ──────────────────────────────

    # 1 뉴스 실시간 수집 실행 함수
    news_crawled = fetch_latest_news()

    if len(news_crawled) != 0:
        # title 중복 제거
        news_crawled = get_news_deduplicate_by_title(news_crawled)

        save_to_db(news_crawled)

        send_to_redis(news_crawled)

    # ──────────────────────────────
    # 2 뉴스 전처리
    # - 본문 전처리 및 요약, 종목과 업종명 매칭
    # ──────────────────────────────

    # 1 뉴스 본문 전처리 실행 함수
    filtered_news = []

    if len(news_crawled) != 0:
        for news in news_crawled:
            news_article = news["article"]
            news_article_preprocessed = remove_market_related_sentences(news_article)

            if len(news_article_preprocessed) < 70:
                continue  # 본문 길이 짧으면 제외

            news["article_preprocessed"] = news_article_preprocessed
            filtered_news.append(news)

    print(f"\n필터링된 뉴스 {filtered_news}\n")

    # 2 뉴스 본문 요악 함수
    summarzied_news = []

    if len(filtered_news) != 0:
        for news in filtered_news:
            news_article = news["article_preprocessed"]
            summary = get_article_summary(news_article)

            if len(summary) < 70:
                continue  # 본문 길이 짧으면 제외

            news["summary"] = summary
            summarzied_news.append(news)

    print(f"\n요약된 뉴스  {summarzied_news}\n")

    ner_news = []

    # 3 뉴스 종목, 업종명 매칭 함수
    if len(summarzied_news) != 0:
        for news in summarzied_news:
            news_summary = news["summary"]
            stock_list = get_stock_list(news_summary)

            # 여기서 필터링
            stock_list = filter_official_stocks_from_list(
                stock_list, official_stock_set
            )
            news["stock_list"] = stock_list

            # 종목 없거나 너무 많으면 제외
            if len(stock_list) > 4 or len(stock_list) < 1:
                news["stock_list"] = None

            industry_list = get_industry_list_from_stocks(stock_list, stock_to_industry)
            news["industry_list"] = industry_list

            if len(industry_list) < 1:
                news["industry_list"] = None

            ner_news.append(news)

            # 중복 뉴스 제거
            ner_news = get_news_deduplicate_by_title(ner_news)

    print(f"\n종목, 업종명 매칭 뉴스 {ner_news}\n")

    save_to_db_metadata(ner_news)

    # ──────────────────────────────
    # 3 뉴스 경제 및 행동 지표 피쳐 추가
    # - 주가 D+1~D+30 변동률, 금리, 환율, 기관 매매동향, 유가 등
    # ──────────────────────────────

    if len(ner_news) != 0:
        news_list = ner_news
        pipeline = NewsMarketPipeline(news_list=news_list, df_base_rate=df_base_rate)

        market_datas = pipeline.run()

        if market_datas:
            pass

    market_datas = [
        {
            "news_id": "2023555_11",
            "d_minus_5_date_close": 56800.0,
            "d_minus_5_date_volume": 12870515.0,
            "d_minus_5_date_foreign": 49110828550,
            "d_minus_5_date_institution": -33974732600,
            "d_minus_5_date_individual": -13745996750,
            "d_minus_4_date_close": 57800.0,
            "d_minus_4_date_volume": 19649983.0,
            "d_minus_4_date_foreign": 119775101650,
            "d_minus_4_date_institution": 67064831850,
            "d_minus_4_date_individual": -176290654250,
            "d_minus_3_date_close": 59100.0,
            "d_minus_3_date_volume": 23266027.0,
            "d_minus_3_date_foreign": 281318315200,
            "d_minus_3_date_institution": 137802524700,
            "d_minus_3_date_individual": -414294844450,
            "d_minus_2_date_close": 59800.0,
            "d_minus_2_date_volume": 19609659.0,
            "d_minus_2_date_foreign": 309135095150,
            "d_minus_2_date_institution": -129336955150,
            "d_minus_2_date_individual": -159736050350,
            "d_minus_1_date_close": 59200.0,
            "d_minus_1_date_volume": 15305760.0,
            "d_minus_1_date_foreign": -54533069200,
            "d_minus_1_date_institution": -17931077300,
            "d_minus_1_date_individual": 71865041600,
            "fx": 1359.9,
            "bond10y": 2.419,
            "base_rate": 2.5,
        },
        {
            "news_id": "2023555_11",
            "d_minus_5_date_close": 57800.0,
            "d_minus_5_date_volume": 19649983.0,
            "d_minus_5_date_foreign": 119775101650,
            "d_minus_5_date_institution": 67064831850,
            "d_minus_5_date_individual": -176290654250,
            "d_minus_4_date_close": 59100.0,
            "d_minus_4_date_volume": 23266027.0,
            "d_minus_4_date_foreign": 281318315200,
            "d_minus_4_date_institution": 137802524700,
            "d_minus_4_date_individual": -414294844450,
            "d_minus_3_date_close": 59800.0,
            "d_minus_3_date_volume": 19609659.0,
            "d_minus_3_date_foreign": 309135095150,
            "d_minus_3_date_institution": -129336955150,
            "d_minus_3_date_individual": -159736050350,
            "d_minus_2_date_close": 59200.0,
            "d_minus_2_date_volume": 15305760.0,
            "d_minus_2_date_foreign": -54533069200,
            "d_minus_2_date_institution": -17931077300,
            "d_minus_2_date_individual": 71865041600,
            "d_minus_1_date_close": 59900.0,
            "d_minus_1_date_volume": 13610734.0,
            "d_minus_1_date_foreign": 133535624000,
            "d_minus_1_date_institution": 24499766250,
            "d_minus_1_date_individual": -148212988300,
            "fx": 1359.9,
            "bond10y": 2.419,
            "base_rate": 2.5,
        },
    ]
    save_to_db_external(market_datas)

    # ──────────────────────────────
    # 4 뉴스 시멘틱 피쳐 추가
    # - topic별 분포값, 클러스터 동일 여부
    # ──────────────────────────────

    # 으악


if __name__ == "__main__":
    # 데이터 불러오기
    # 현재 스크립트 기준 디렉토리 (automation/scripts/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    official_stock_path = os.path.abspath(
        os.path.join(BASE_DIR, "../db/KRX_KOSPI_STOCK.csv")
    )
    official_stock_set = load_official_stock_list(official_stock_path)

    industry_map_path = os.path.abspath(
        os.path.join(BASE_DIR, "../db/KRX_KOSPI_DESCRIPTION.csv")
    )
    stock_to_industry = load_stock_to_industry_map(industry_map_path)

    korea_base_rate_daily_path = os.path.abspath(
        os.path.join(BASE_DIR, "../db/korea_base_rate_daily.csv")
    )
    df_base_rate = load_rate_df(korea_base_rate_daily_path)

    log.info("✅ run_scheduler.py 시작됨")

    # 첫 실행 즉시
    job(official_stock_set, stock_to_industry, df_base_rate)

    # 이후 매 1분마다 실행
    schedule.every(1).minutes.do(
        lambda: job(official_stock_set, stock_to_industry, df_base_rate)
    )

    while True:
        schedule.run_pending()
        time.sleep(1)
