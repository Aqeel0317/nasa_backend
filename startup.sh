#!/bin/sh
# Azure App Service start command. App Service routes traffic to $PORT (default 8000).
exec gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 2 --bind 0.0.0.0:${PORT:-8000} --timeout 180
