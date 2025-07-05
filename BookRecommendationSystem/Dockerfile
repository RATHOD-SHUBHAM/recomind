# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy pyproject.toml and uv.lock first for better caching
COPY pyproject.toml uv.lock* ./

# Install Python dependencies using uv (without installing the package itself)
RUN uv pip install --system pandas matplotlib seaborn python-dotenv langchain langchain-core langchain-community langchain-text-splitters langchain-openai langchain-chroma transformers streamlit protobuf chromadb openai

# Copy application code
COPY . .

# Create directory for vector database and set permissions
RUN mkdir -p vector_db && chmod 777 vector_db

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application using uv
CMD ["uv", "run", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"] 