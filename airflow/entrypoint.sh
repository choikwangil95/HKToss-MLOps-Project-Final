#!/bin/bash
set -e

echo "Fixing log directory permission..."
mkdir -p /opt/airflow/logs

# 권한 변경은 root 사용자만 가능하므로 실패해도 무시
chown -R airflow:root /opt/airflow/logs || true
chmod -R 755 /opt/airflow/logs || true

echo "Installing requirements..."
if [ -f "/requirements.txt" ]; then
  pip install -r /requirements.txt || echo "⚠️ Failed to install requirements"
else
  echo "⚠️ /requirements.txt not found, skipping installation"
fi

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
