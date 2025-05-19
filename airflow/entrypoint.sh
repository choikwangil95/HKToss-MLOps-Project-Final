#!/bin/bash
set -e

echo "🔒 Fixing log directory permission..."
mkdir -p /opt/airflow/logs
chown -R airflow:root /opt/airflow/logs || true
chmod -R 755 /opt/airflow/logs || true

echo "📦 Installing requirements..."
pip install -r /requirements.txt

if [ "$1" = "webserver" ]; then
  echo "🔧 Initializing Airflow DB..."
  airflow db upgrade

  echo "👤 Creating admin user..."
  airflow users create \
    --username admin \
    --password admin \
    --firstname Air \
    --lastname Flow \
    --role Admin \
    --email admin@example.com || true
fi

echo "🚀 Starting Airflow with command: $@"
exec airflow "$@"
