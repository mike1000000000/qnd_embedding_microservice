FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pull Model
ARG MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV MODEL=${MODEL}

# Download and cache the default model during the build
RUN python -c "from sentence_transformers import SentenceTransformer; \
                model = SentenceTransformer('$MODEL'); \
                model.save('/app/model/$MODEL')"

COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

