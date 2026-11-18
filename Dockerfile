FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the HuggingFaceEmbeddings model to avoid downloading it on the first request
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

COPY app/ ./app/
COPY ingest.py .
COPY documents/ ./documents/

# bake the ingest step into the image so the first request does not have to do it
RUN python ingest.py

# The app sits behind a reverse proxy (Hugging Face's on the Space), so the TCP peer
# is the proxy, not the visitor. uvicorn reads the visitor's address from
# X-Forwarded-For, but only when the peer is one of these private ranges, and it
# reads the header from the right, skipping trusted hops, so an address a client
# writes into the header itself is never picked. Override on the host if needed.
ENV FORWARDED_ALLOW_IPS="127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.64.0.0/10,fc00::/7"

EXPOSE 8000
# --no-access-log: the app logs each request itself, with latency and without raw IPs
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-access-log"]