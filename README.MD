# Sentence Embedding Microservice

This microservice uses the **Sentence-Transformers** library to generate sentence embeddings. It is designed to provide a simple API endpoint that accepts text input and returns its corresponding embedding.

The service is built as a Docker container, making it easy to deploy and use in any environment. It downloads and saves a pre-trained sentence transformer model (`all-MiniLM-L6-v2`) and serves it through a REST API using **FastAPI**.

## Features

- **Text Embedding Generation**: Accepts a text input and generates its embedding using the `all-MiniLM-L6-v2` model from the **Sentence-Transformers** library.
- **Dockerized**: The service is containerized using Docker for easy deployment.
- **GPU Support**: The service supports CUDA for faster computations if available (via `torch`).
- **API Key Authentication**: Protects the endpoint with an API key to ensure secure access.

## Requirements

- **Python 3.8+** (for running the app manually)
- **Docker** (for containerization)
- **Uvicorn** (for running the app manually)

## Setup and Usage

Follow these steps to build and run the microservice:

### 1. Clone the Repository

Clone the repository containing the `Dockerfile` and the application code:

```bash
git clone https://github.com/mike1000000000/qnd_embedding_microservice.git
cd qnd_embedding_microservice
```

### 2. Generate or Set the API Key

The service uses an API key for authentication. On the first run, the app will generate an `API_KEY` and save it to a `.env` file. The key will also be displayed in the logs. Make sure to copy and store this key securely. You can also manually set an API key by adding it to the `.env` file:

```env
API_KEY=your_generated_api_key
```

### 3. Run the App Manually

To run the app without Docker, use the following steps:

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app using Uvicorn:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5000
   ```

   The app will now be available on `http://localhost:5000`. 

3. Use the API Key:
   - Include the `X-API-Key` header in all requests to authenticate. For example:

     ```bash
     curl -X POST http://localhost:5000/embeddings \
         -H "Content-Type: application/json" \
         -H "X-API-Key: your_api_key_here" \
         -d '{"text": "Hello, this is an example."}'
     ```

### 4. Build and Run the Docker Container

Alternatively, you can build and run the service as a Docker container:

#### Build the Docker Image

```bash
docker build -t sentence-embedding-service .
```

#### Run the Docker Container

```bash
docker run -p 5000:5000 sentence-embedding-service
```

The service will now be available on port `5000`. You can adjust the port mapping as needed.

#### Access the API Documentation

The microservice provides interactive API documentation via **FastAPI**. After starting the service, you can access it at:

- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## Troubleshooting

- **Docker container fails to start**:
  - Ensure that Docker is running and has sufficient resources allocated.
  - Make sure the `model` directory is accessible within the container.

- **Missing Model Error**:
  - If the model isn't downloaded properly, try rebuilding the Docker image to ensure all steps are executed correctly.

- **Invalid API Key**:
  - If you see a `401 Unauthorized` error, ensure that the `X-API-Key` header matches the key in the `.env` file.

## License

This microservice is open-source and available for use under the MIT License.