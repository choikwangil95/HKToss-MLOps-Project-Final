FROM python:3.10-slim

# 2. 시스템 패키지 설치 (OpenJDK는 default-jdk로 교체 권장)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    gcc \
    g++ \
    libpq-dev \
    build-essential \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:$PATH"

# 3. 최신 pip 설치
RUN pip install --upgrade pip

# 4. 의존성 먼저 복사 (캐시 최적화)
COPY requirements.txt .

RUN pip install -r requirements.txt

# 5. 앱 코드 복사
WORKDIR /modelapi2
COPY . .

# 6. 포트 오픈
EXPOSE 80

# 7. 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
