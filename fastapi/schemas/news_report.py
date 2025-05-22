from pydantic import BaseModel
from typing import List
from datetime import date

class NewsIDRequest(BaseModel):
    news_id: int

class PastReportItem(BaseModel):
    matched_news_title: str     
    matched_news_date: date    
    report_title: str            
    report_content: str    
    report_date: date            
    similarity: float         

class PastReportsResponse(BaseModel):
    results: List[PastReportItem]