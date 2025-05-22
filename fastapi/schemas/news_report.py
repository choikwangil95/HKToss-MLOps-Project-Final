from pydantic import BaseModel
from typing import List , Union
from datetime import date

class NewsIDRequest(BaseModel):
    news_id: int

class PastReportItem(BaseModel):
    report_title: str
    report_content: str
    report_date: Union[date, str]
    company: str
    target_price: str
    opinion: str
    similarity: float   

class PastReportsResponse(BaseModel):
    results: List[PastReportItem]