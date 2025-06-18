from sqlalchemy.orm import Session
from models.custom import (
    NewsModel_v2,
    NewsModel_v2_External,
    NewsModel_v2_Metadata,
)
import pandas as pd
import numpy as np
from fastapi import Request
from services.model import get_news_embeddings


def create_engineered_features_complete(df: pd.DataFrame) -> pd.DataFrame:
    """특성공학 함수 - 첨부된 코드와 동일"""
    df = df.copy()
    print("특성공학 시작...")
    print(f"특성공학 전 컬럼 수: {len(df.columns)}")

    # 1. 일별 수익률 계산
    daily_return_cols = []
    for i in range(1, 5):
        current_close_key = f"d_minus_{i}_date_close"
        prev_close_key = f"d_minus_{i+1}_date_close"
        return_key = f"d_minus_{i}_daily_return"

        if current_close_key in df.columns and prev_close_key in df.columns:
            df[return_key] = (df[current_close_key] - df[prev_close_key]) / df[
                prev_close_key
            ].replace(0, np.nan)
            df[return_key] = df[return_key].fillna(0)
            daily_return_cols.append(return_key)
            print(f"  생성: {return_key}")

    # 2. 이동평균 계산
    moving_average_cols = []
    if all(
        col in df.columns
        for col in [
            "d_minus_1_date_close",
            "d_minus_2_date_close",
            "d_minus_3_date_close",
        ]
    ):
        df["ma_3_d_minus_1"] = df[
            ["d_minus_1_date_close", "d_minus_2_date_close", "d_minus_3_date_close"]
        ].mean(axis=1)
        moving_average_cols.append("ma_3_d_minus_1")
        print(f"  생성: ma_3_d_minus_1")

    if all(
        col in df.columns
        for col in [
            "d_minus_1_date_close",
            "d_minus_2_date_close",
            "d_minus_3_date_close",
            "d_minus_4_date_close",
            "d_minus_5_date_close",
        ]
    ):
        df["ma_5_d_minus_1"] = df[
            [
                "d_minus_1_date_close",
                "d_minus_2_date_close",
                "d_minus_3_date_close",
                "d_minus_4_date_close",
                "d_minus_5_date_close",
            ]
        ].mean(axis=1)
        moving_average_cols.append("ma_5_d_minus_1")
        print(f"  생성: ma_5_d_minus_1")

    # 3. 변동성 계산
    volatility_cols = []
    if len(daily_return_cols) >= 2:
        df["vol_5_d_minus_1"] = df[daily_return_cols].std(axis=1)
        df["vol_5_d_minus_1"] = df["vol_5_d_minus_1"].fillna(0)
        volatility_cols.append("vol_5_d_minus_1")
        print(f"  생성: vol_5_d_minus_1")

    print(f"특성공학 후 컬럼 수: {len(df.columns)}")
    return df


def get_complete_external_cols():
    """외부 특성 컬럼 순서 정의 - 첨부된 코드와 동일"""
    group_macro = ["fx", "bond10y", "base_rate"]  # 3개
    group_volume = [
        "d_minus_5_date_volume",
        "d_minus_4_date_volume",
        "d_minus_3_date_volume",
        "d_minus_2_date_volume",
        "d_minus_1_date_volume",
    ]  # 5개
    group_foreign = [
        "d_minus_5_date_foreign",
        "d_minus_4_date_foreign",
        "d_minus_3_date_foreign",
        "d_minus_2_date_foreign",
        "d_minus_1_date_foreign",
    ]  # 5개
    group_institution = [
        "d_minus_5_date_institution",
        "d_minus_4_date_institution",
        "d_minus_3_date_institution",
        "d_minus_2_date_institution",
        "d_minus_1_date_institution",
    ]  # 5개
    group_individual = [
        "d_minus_5_date_individual",
        "d_minus_4_date_individual",
        "d_minus_3_date_individual",
        "d_minus_2_date_individual",
        "d_minus_1_date_individual",
    ]  # 5개
    group_price_close = [
        "d_minus_5_date_close",
        "d_minus_4_date_close",
        "d_minus_3_date_close",
        "d_minus_2_date_close",
        "d_minus_1_date_close",
    ]  # 5개
    daily_return_cols = [
        "d_minus_1_daily_return",
        "d_minus_2_daily_return",
        "d_minus_3_daily_return",
        "d_minus_4_daily_return",
    ]  # 4개
    moving_average_cols = ["ma_3_d_minus_1", "ma_5_d_minus_1"]  # 2개
    volatility_cols = ["vol_5_d_minus_1"]  # 1개

    groups = [
        group_macro,  # 0
        group_volume,  # 1
        group_foreign,  # 2
        group_institution,  # 3
        group_individual,  # 4
        group_price_close,  # 5
        daily_return_cols,  # 6
        moving_average_cols,  # 7
        volatility_cols,  # 8
    ]

    external_cols = []
    for group in groups:
        external_cols.extend(group)

    return external_cols, groups


async def get_embedding(text: str, request) -> np.ndarray:
    """API 대신 내부 임베딩 함수 사용 (실패 시에만 디버깅)"""
    if not text or len(text.strip()) == 0:
        raise ValueError("임베딩할 텍스트가 비어있습니다.")

    try:
        embedding_list = get_news_embedding(text.strip(), request)
        embedding = np.array(embedding_list, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        return embedding

    except Exception as e:
        print(f"임베딩 벡터 생성 실패: {e}")
        raise RuntimeError(f"임베딩 벡터 생성 실패: {e}")


def apply_scaling_complete(df: pd.DataFrame, fitted_scalers: dict) -> tuple:
    """스케일링 적용 - 첨부된 코드와 동일"""
    df_scaled = df.copy()
    external_cols, groups = get_complete_external_cols()

    print(f"스케일링 시작... 기대 컬럼 수: {len(external_cols)}")

    for group_idx, group_cols in enumerate(groups):
        if group_idx in fitted_scalers:
            scaler = fitted_scalers[group_idx]
            valid_cols = [col for col in group_cols if col in df.columns]
            missing_cols = [col for col in group_cols if col not in df.columns]

            if missing_cols:
                print(f"그룹 {group_idx} 누락 컬럼: {missing_cols}")

            if valid_cols:
                df_scaled[valid_cols] = scaler.transform(df[valid_cols])
                print(
                    f"그룹 {group_idx}: {len(valid_cols)}/{len(group_cols)}개 컬럼 스케일링 완료"
                )

    return df_scaled, external_cols


def get_news_data_from_db(news_id: str, db: Session) -> dict:
    """DB에서 뉴스 관련 모든 데이터 조회 - 실제 모델 사용"""[5]
    print(f"DB에서 뉴스 데이터 조회: {news_id}")

    # 3개 테이블 조인하여 모든 데이터 한 번에 조회
    query_result = (
        db.query(
            NewsModel_v2.news_id,
            NewsModel_v2.title,
            NewsModel_v2.wdate,
            NewsModel_v2_Metadata.summary,
            NewsModel_v2_Metadata.stock_list,
            NewsModel_v2_External.d_minus_5_date_close,
            NewsModel_v2_External.d_minus_4_date_close,
            NewsModel_v2_External.d_minus_3_date_close,
            NewsModel_v2_External.d_minus_2_date_close,
            NewsModel_v2_External.d_minus_1_date_close,
            NewsModel_v2_External.d_minus_5_date_volume,
            NewsModel_v2_External.d_minus_4_date_volume,
            NewsModel_v2_External.d_minus_3_date_volume,
            NewsModel_v2_External.d_minus_2_date_volume,
            NewsModel_v2_External.d_minus_1_date_volume,
            NewsModel_v2_External.d_minus_5_date_foreign,
            NewsModel_v2_External.d_minus_4_date_foreign,
            NewsModel_v2_External.d_minus_3_date_foreign,
            NewsModel_v2_External.d_minus_2_date_foreign,
            NewsModel_v2_External.d_minus_1_date_foreign,
            NewsModel_v2_External.d_minus_5_date_institution,
            NewsModel_v2_External.d_minus_4_date_institution,
            NewsModel_v2_External.d_minus_3_date_institution,
            NewsModel_v2_External.d_minus_2_date_institution,
            NewsModel_v2_External.d_minus_1_date_institution,
            NewsModel_v2_External.d_minus_5_date_individual,
            NewsModel_v2_External.d_minus_4_date_individual,
            NewsModel_v2_External.d_minus_3_date_individual,
            NewsModel_v2_External.d_minus_2_date_individual,
            NewsModel_v2_External.d_minus_1_date_individual,
            NewsModel_v2_External.fx,
            NewsModel_v2_External.bond10y,
            NewsModel_v2_External.base_rate,
        )
        .join(
            NewsModel_v2_Metadata, NewsModel_v2.news_id == NewsModel_v2_Metadata.news_id
        )
        .join(
            NewsModel_v2_External, NewsModel_v2.news_id == NewsModel_v2_External.news_id
        )
        .filter(NewsModel_v2.news_id == news_id)
        .first()
    )

    if not query_result:
        raise ValueError(
            f"뉴스 ID {news_id}를 찾을 수 없거나 필요한 데이터가 누락되었습니다."
        )

    # summary 검증 강화
    summary = query_result.summary
    if not summary or len(summary.strip()) == 0:
        raise ValueError(f"뉴스 ID {news_id}의 요약 정보가 비어있습니다.")

    print(f"Summary 길이: {len(summary)}")
    print(f"Summary 내용: {summary[:100]}...")  # 처음 100자만 출력

    # 결과를 dict로 변환
    result = {
        "news_id": query_result.news_id,
        "title": query_result.title,
        "summary": query_result.summary,
        "wdate": query_result.wdate,
        "stock_list": query_result.stock_list,
        # 주가 데이터
        "d_minus_5_date_close": float(query_result.d_minus_5_date_close or 0),
        "d_minus_4_date_close": float(query_result.d_minus_4_date_close or 0),
        "d_minus_3_date_close": float(query_result.d_minus_3_date_close or 0),
        "d_minus_2_date_close": float(query_result.d_minus_2_date_close or 0),
        "d_minus_1_date_close": float(query_result.d_minus_1_date_close or 0),
        # 거래량 데이터
        "d_minus_5_date_volume": float(query_result.d_minus_5_date_volume or 0),
        "d_minus_4_date_volume": float(query_result.d_minus_4_date_volume or 0),
        "d_minus_3_date_volume": float(query_result.d_minus_3_date_volume or 0),
        "d_minus_2_date_volume": float(query_result.d_minus_2_date_volume or 0),
        "d_minus_1_date_volume": float(query_result.d_minus_1_date_volume or 0),
        # 투자자별 순매수 데이터
        "d_minus_5_date_foreign": float(query_result.d_minus_5_date_foreign or 0),
        "d_minus_4_date_foreign": float(query_result.d_minus_4_date_foreign or 0),
        "d_minus_3_date_foreign": float(query_result.d_minus_3_date_foreign or 0),
        "d_minus_2_date_foreign": float(query_result.d_minus_2_date_foreign or 0),
        "d_minus_1_date_foreign": float(query_result.d_minus_1_date_foreign or 0),
        "d_minus_5_date_institution": float(
            query_result.d_minus_5_date_institution or 0
        ),
        "d_minus_4_date_institution": float(
            query_result.d_minus_4_date_institution or 0
        ),
        "d_minus_3_date_institution": float(
            query_result.d_minus_3_date_institution or 0
        ),
        "d_minus_2_date_institution": float(
            query_result.d_minus_2_date_institution or 0
        ),
        "d_minus_1_date_institution": float(
            query_result.d_minus_1_date_institution or 0
        ),
        "d_minus_5_date_individual": float(query_result.d_minus_5_date_individual or 0),
        "d_minus_4_date_individual": float(query_result.d_minus_4_date_individual or 0),
        "d_minus_3_date_individual": float(query_result.d_minus_3_date_individual or 0),
        "d_minus_2_date_individual": float(query_result.d_minus_2_date_individual or 0),
        "d_minus_1_date_individual": float(query_result.d_minus_1_date_individual or 0),
        # 매크로 데이터
        "fx": float(query_result.fx or 0),
        "bond10y": float(query_result.bond10y or 0),
        "base_rate": float(query_result.base_rate or 0),
    }

    print(f"DB 데이터 조회 완료: {len(result)} 필드")
    return result


async def predict_and_calculate_impact(data: dict, request: Request):
    sess = request.app.state.predictor
    target_scaler = request.app.state.target_scaler
    fitted_scalers = request.app.state.group_scalers

    print("=== 예측 시작 ===")
    df = pd.DataFrame([data])
    print(f"입력 데이터 shape: {df.shape}")

    # 2. 특성공학 적용
    df_engineered = create_engineered_features_complete(df)

    # 3. 스케일링 적용
    df_scaled, external_cols = apply_scaling_complete(df_engineered, fitted_scalers)

    # 4. 첫 번째 행 추출
    row = df_scaled.iloc[0]

    # 5. 임베딩 생성
    embedding = await get_news_embeddings([row["summary"]], request)
    embedding = np.array([embedding[0]], dtype=np.float32)  # ✅ 2D로 reshape
    print(f"임베딩 형태: {embedding.shape}")

    # 6. 외부 특성 벡터 생성
    external_vector = []
    missing_cols = []
    for col in external_cols:
        if col in row.index:
            external_vector.append(row[col])
        else:
            external_vector.append(0.0)
            missing_cols.append(col)
    if missing_cols:
        print(f"누락된 컬럼들 (0으로 채움): {missing_cols}")

    external_array = np.array([external_vector], dtype=np.float32)
    print(f"외부 특성 형태: {external_array.shape}")

    # 7. ONNX 추론
    inputs = {"embedding": embedding, "external": external_array}
    raw_predictions = sess.run(None, inputs)[0]
    print(f"ONNX 원시 출력: {raw_predictions}")

    # 8. 역스케일링
    predictions_original = target_scaler.inverse_transform(raw_predictions)
    predicted_closes = predictions_original[0].tolist()
    print(f"예측된 종가 (d+1~d+5): {predicted_closes}")

    baseline_mean = [-0.0078573, -0.0083497, -0.00810895, -0.00799404, -0.00891119]
    baseline_std = [0.06588189, 0.07283064, 0.07710478, 0.08238087, 0.08863377]

    z_scores = (np.array(predicted_closes) - np.array(baseline_mean)) / np.array(
        baseline_std
    )

    # 9. 과거 종가 추출
    historical_closes = [
        data["d_minus_1_date_close"],
        data["d_minus_2_date_close"],
        data["d_minus_3_date_close"],
        data["d_minus_4_date_close"],
        data["d_minus_5_date_close"],
    ]
    print(f"과거 종가 (d-1~d-5): {historical_closes}")

    # 10. impact_score 계산
    all_closes = historical_closes + predicted_closes
    max_price = max(all_closes)
    min_price = min(all_closes)
    impact_score = round(abs(max_price - min_price), 2)

    print(f"전체 가격 범위: {min_price:.2f} ~ {max_price:.2f}")
    print(f"최종 impact_score: {impact_score:.6f}")

    return predicted_closes, historical_closes, impact_score, z_scores


async def get_news_impact_score_service(news_id: str, db: Session, request: Request):
    """메인 서비스 함수: 뉴스 ID로 임팩트 스코어 계산"""
    print(f"=== 서비스 시작: news_id={news_id} ===")

    try:
        # 1. DB에서 데이터 조회
        data = get_news_data_from_db(news_id, db)
        print(f"✓ DB 데이터 조회 완료")

        # 2. 예측 및 임팩트 스코어 계산
        d_plus, d_minus, impact_score, z_scores = await predict_and_calculate_impact(
            data, request
        )
        print(f"✓ 임팩트 스코어 계산 완료: {impact_score}")

        return d_plus, d_minus, impact_score, z_scores

    except Exception as e:
        print(f"서비스 오류: {e}")
        raise e
