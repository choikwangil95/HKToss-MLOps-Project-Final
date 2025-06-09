FROM python:3.10-slim

# 1. 작업 디렉토리 설정
WORKDIR /automation

# 2. 의존성만 먼저 복사
COPY requirements.txt .
COPY requirements-dev.txt .

# 3. 패키지 설치 (캐시 최적화)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# 4. 전체 코드 복사
COPY . .

# 5. 실행 명령어
CMD ["python", "scripts/run_scheduler.py"]
