#!/bin/bash
set -e

echo "✅ Installing requirements..."
pip install -r /requirements.txt

echo "🔧 Initializing Airflow DB (only on webserver)..."
if [ "$1" = "webserver" ]; then
    airflow db upgrade

    echo "👤 Creating admin user (if not exists)..."
    airflow users create \
        --username admin \
        --password admin \
        --firstname Air \
        --lastname Flow \
        --role Admin \
        --email admin@example.com || true
fi

echo "🚀 Starting Airflow: $@"
exec airflow "$@"
