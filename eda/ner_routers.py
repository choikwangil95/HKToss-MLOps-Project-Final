from fastapi import APIRouter, HTTPException
from schemas import StockPredictByFieldRequest, StockPredictResponse
from services import extract_stock_labels_by_label_id

router = APIRouter()

@router.post(
    '/stocks/predict-by-fields',
    response_model=StockPredictResponse,
    summary='제목 + 본문 기반 종목명 예측',
    description='뉴스 제목과 본문을 각각 받아 종합 분석하여 종목명을 추출합니다.',
)
def predict_stocks_from_fields(req: StockPredictByFieldRequest):
    full_text = f'{req.title.strip()} {req.content.strip()}'
    try:
        stocks = extract_stock_labels_by_label_id(full_text)
        return {'stocks': stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
