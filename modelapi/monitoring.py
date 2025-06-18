import os
from typing import Callable

import numpy as np

# prometheus_client : 프로메테우스에서 파이썬 기반으로 작동할 수 있게 해주는 패키지
from prometheus_client import Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

# METRICS_NAMESPACE가 없으면 "fastapi"로 새롭게 생성, 있으면 "fastapi" 값을 가져옴
NAMESPACE = os.environ.get("METRICS_NAMESPACE", "fastapi")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    # endpoint가 /metrics로 노출 됨
    excluded_handlers=["/metrics"],
    # 환경 변수( 도커 또는 도커 컴포즈)가 ENABLE_METRICS가 true인 경우만 작동됨
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)


# CUSTOM METRIC


# ✅ 전역 Gauge 정의 (최근 값 저장용)
LATEST_SCORE = Gauge(
    name="latest_news_impact_score",
    documentation="Latest news impact score from /news/{news_id}/impact_score",
    namespace="fastapi_model",
    subsystem="news",
)


def regression_model_output(
    metric_name: str = "news_impact_score_output",
    metric_doc: str = "Impact score value from /news/{news_id}/impact_score",
    metric_namespace: str = "",
    metric_subsystem: str = "",
    buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")),
) -> Callable[[Info], None]:
    METRIC = Histogram(
        metric_name,
        metric_doc,
        buckets=buckets,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler.startswith(
            "/news/"
        ) and info.modified_handler.endswith("/impact_score"):
            score = info.response.headers.get("X-model-score")
            if score:
                try:
                    score_val = float(score)
                    METRIC.observe(score_val)  # ✅ 분포 기록
                    LATEST_SCORE.set(score_val)  # ✅ 최근 값 기록
                except ValueError:
                    pass

    return instrumentation


buckets = (*np.arange(0, 10.5, 0.5).tolist(), float("inf"))
instrumentator.add(
    regression_model_output(
        metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM, buckets=buckets
    )
)
