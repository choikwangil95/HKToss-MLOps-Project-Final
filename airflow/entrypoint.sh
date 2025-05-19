#!/bin/bash
set -e

echo "Installing requirements..."
if [ -f "/requirements.txt" ]; then
  pip install -r /requirements.txt || echo "⚠️ Failed to install requirements"
else
  echo "⚠️ /requirements.txt not found, skipping installation"
fi

# 로그 디렉토리 생성 및 퍼미션 설정
mkdir -p /opt/airflow/logs/scheduler
chmod -R 777 /opt/airflow/logs

if [ "$1" = "webserver" ]; then
  echo "Initializing Airflow DB..."
  airflow db upgrade

  echo "Creating admin user..."
  airflow users create \
    --username admin \
    --password admin \
    --firstname Air \
    --lastname Flow \
    --role Admin \
    --email admin@example.com || true
fi

echo "Starting Airflow with command: $@"
exec airflow "$@"
