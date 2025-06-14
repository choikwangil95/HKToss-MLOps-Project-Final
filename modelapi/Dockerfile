FROM python:3.10-slim

# 필수 시스템 패키지 설치 (특히 psycopg2에 필요)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 최신 pip 설치
RUN pip install --upgrade pip

COPY requirements.txt requirements.txt 

RUN pip install -r requirements.txt

WORKDIR /modelapi

COPY . /modelapi

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
