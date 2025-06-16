FROM python:3.11-slim

WORKDIR /app

# 시스템 종속성 설치 (Faiss, numpy 등 빌드용)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt


COPY . .

CMD ["uvicorn", "fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
