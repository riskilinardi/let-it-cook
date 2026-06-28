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

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]