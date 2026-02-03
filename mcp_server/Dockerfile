FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY finance_server.py .

# Expose SSE port
EXPOSE 8080

# Run the server
CMD ["python", "finance_server.py"]
