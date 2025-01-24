# Sentence Embedding Microservice

This microservice uses the **Sentence-Transformers** library to generate sentence embeddings. It is designed to provide a simple API endpoint that accepts text input and returns its corresponding embedding.

The service is built as a Docker container, making it easy to deploy and use in any environment. It downloads and saves a pre-trained sentence transformer model (`all-MiniLM-L6-v2`) and serves it through a REST API using **FastAPI**.

## Features

- **Text Embedding Generation**: Accepts a text input and generates its embedding using the `all-MiniLM-L6-v2` model from the **Sentence-Transformers** library.
- **Dockerized**: The service is containerized using Docker for easy deployment.
- **GPU Support**: The service supports CUDA for faster computations if available (via `torch`).

## Requirements

- **Docker** (for containerization)

## Setup and Usage

Follow these steps to build and run the microservice:

### 1. Clone the Repository

Clone the repository containing the `Dockerfile` and the application code:

```bash
git clone https://github.com/mike1000000000/qnd_embedding_microservice.git
cd qnd_embedding_microservice
```

### 2. Build the Docker Image

Build the Docker image for the microservice:

```bash
docker build -t sentence-embedding-service .
```

This step will:

1. Install the necessary Python dependencies.
2. Download and save the `all-MiniLM-L6-v2` model to the `model/` directory inside the Docker container.

### 3. Run the Docker Container

Start the microservice by running the Docker container:

```bash
docker run -p 5000:5000 sentence-embedding-service
```

The service will now be available on port `5000`. You can adjust the port mapping as needed.

### 4. Send a Request to the API

To get an embedding for a given sentence, make a POST request to `http://localhost:5000/embeddings`.

Example request using `curl`:

```bash
curl -X POST http://localhost:5000/embeddings \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, this is an example."}'
```

The response will look something like this:

```json
{
  "embedding": [0.1, 0.2, 0.3, ...]  // A list of floats representing the sentence embedding.
}
```

### 5. Access the API Documentation

The microservice provides interactive API documentation via **FastAPI**. After starting the service, you can access it at:

- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## Troubleshooting

- **Docker container fails to start**:
  - Ensure that Docker is running and has sufficient resources allocated. **Note**: This container can be 6GB.

- **Missing Model Error**:
  - If the model isn't downloaded properly, try rebuilding the Docker image to ensure all steps are executed correctly.

## License

This microservice is open-source and available for use under the MIT License.

