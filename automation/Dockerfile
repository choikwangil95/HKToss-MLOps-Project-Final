FROM python:3.10-slim

# 1. 작업 디렉토리 설정
WORKDIR /automation

# 2. 먼저 requirements.txt만 복사
COPY requirements.txt .
COPY requirements-dev.txt .

# 3. pip install을 먼저 수행 (요게 캐시 핵심!)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

# 4. 나머지 코드 복사
COPY . .

# 5. 실행 커맨드
CMD ["python", "scripts/run_scheduler.py"]
