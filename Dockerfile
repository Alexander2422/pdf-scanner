# python:3.11-slim has official arm64/armv7 builds, so this same
# Dockerfile works both on your dev machine and on the Pi.
FROM python:3.11-slim

# OpenCV needs a couple of system libs even in "headless" mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Uploaded/output files live here — mounted as a volume in docker-compose
# so they survive container restarts and are easy to inspect from the Pi.
RUN mkdir -p /app/uploads /app/output

EXPOSE 5000

CMD ["python", "main.py"]
