import pandas as pd
import os

def load_company_names():
    path = os.path.join('fastapi', 'datas', 'KRX_기업명_20250521.csv')
    df = pd.read_csv(path)
    return set(df['종목명'].dropna().unique())