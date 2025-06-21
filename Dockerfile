FROM python:3.10.17-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git wget \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY gunicorn.conf.py .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ app/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]
