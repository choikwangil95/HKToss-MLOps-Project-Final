from news_pipeline import (
    enrich_stock_list,
    extract_industries,
    NewsMarketPipeline,
    fetch_latest_news,
    get_article_summary,
    get_impact_score,
    get_lda_topic,
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
    save_to_db_topics,
    send_to_redis,
    update_db_impact_score,
)
import schedule
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("news_logger")


def job(
    official_stock_set,
    stock_name_to_code,
    stock_to_industry,
    code_to_industry,
    df_base_rate,
):
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
            news_article = news["article_preprocessed"][:300]
            summary = get_article_summary(news_article)

            if len(summary) < 30:
                continue  # 본문 길이 짧으면 제외

            news["summary"] = summary
            summarzied_news.append(news)

    print(f"\n요약된 뉴스  {summarzied_news}\n")

    ner_news = []

    # 3 뉴스 종목, 업종명 매칭 함수
    if len(summarzied_news) != 0:
        for news in summarzied_news:
            # stock_list
            news_summary = news["summary"]
            stock_list = get_stock_list(news_summary)
            stock_list = filter_official_stocks_from_list(
                stock_list, official_stock_set
            )
            stock_list = enrich_stock_list(stock_list, stock_name_to_code)
            news["stock_list"] = stock_list

            # stock_list_view
            news_article = news["article"]
            stock_list_view = get_stock_list(news_summary)
            stock_list_view = filter_official_stocks_from_list(
                stock_list_view, official_stock_set
            )
            stock_list_view = enrich_stock_list(stock_list_view, stock_name_to_code)
            news["stock_list_view"] = stock_list_view

            # 종목 없거나 너무 많으면 제외
            if len(stock_list) > 4 or len(stock_list) < 1:
                news["stock_list"] = None

            industry_list = get_industry_list_from_stocks(stock_list, stock_to_industry)
            industry_list = extract_industries(stock_list, code_to_industry)
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
            save_to_db_external(market_datas)

            score_datas = get_impact_score(market_datas)

            update_db_impact_score(score_datas)

    # ──────────────────────────────
    # 4 뉴스 시멘틱 피쳐 추가
    # - topic별 분포값, 클러스터 동일 여부
    # ──────────────────────────────
    topic_news = []
    if len(ner_news) != 0:
        for news in ner_news:
            # stock_list
            news_summary = news["summary"]

            lda_topics = get_lda_topic(news_summary)

            news["topic_1"] = lda_topics["topic_1"]
            news["topic_2"] = lda_topics["topic_2"]
            news["topic_3"] = lda_topics["topic_3"]
            news["topic_4"] = lda_topics["topic_4"]
            news["topic_5"] = lda_topics["topic_5"]
            news["topic_6"] = lda_topics["topic_6"]
            news["topic_7"] = lda_topics["topic_7"]
            news["topic_8"] = lda_topics["topic_8"]
            news["topic_9"] = lda_topics["topic_9"]

            topic_news.append(news)

    save_to_db_topics(topic_news)


if __name__ == "__main__":
    # 데이터 불러오기

    log.info("데이터 로딩 시작")

    # 현재 스크립트 기준 디렉토리 (automation/scripts/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    official_stock_path = os.path.abspath(
        os.path.join(BASE_DIR, "../db/KRX_KOSPI_STOCK.csv")
    )
    official_stock_set, stock_name_to_code = load_official_stock_list(
        official_stock_path
    )

    industry_map_path = os.path.abspath(
        os.path.join(BASE_DIR, "../db/KRX_KOSPI_DESCRIPTION.csv")
    )
    stock_to_industry, code_to_industry = load_stock_to_industry_map(industry_map_path)

    korea_base_rate_daily_path = os.path.abspath(
        os.path.join(BASE_DIR, "../db/korea_base_rate_daily.csv")
    )
    df_base_rate = load_rate_df(korea_base_rate_daily_path)

    log.info("데이터 로딩 완료")

    log.info("✅ run_scheduler.py 시작됨")

    # 첫 실행 즉시
    job(
        official_stock_set,
        stock_name_to_code,
        stock_to_industry,
        code_to_industry,
        df_base_rate,
    )

    # 이후 매 1분마다 실행
    schedule.every(1).minutes.do(
        lambda: job(
            official_stock_set,
            stock_name_to_code,
            stock_to_industry,
            code_to_industry,
            df_base_rate,
        )
    )

    while True:
        schedule.run_pending()
        time.sleep(1)
