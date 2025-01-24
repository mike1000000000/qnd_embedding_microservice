FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Download and save the model during the build
RUN python -c "from sentence_transformers import SentenceTransformer; \
                model = SentenceTransformer('all-MiniLM-L6-v2'); \
                model.save('/app/model/all-MiniLM-L6-v2')"

COPY . /app/

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

