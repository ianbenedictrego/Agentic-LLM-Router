FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached until requirements change
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# main.py does `from graph import workflow`, so the contents of app/ go in the working dir
COPY app/ .

# Run as a non-root user
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
    
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]