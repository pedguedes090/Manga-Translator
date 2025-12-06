# Manga Translator - HuggingFace Spaces Dockerfile
# Uses Python 3.10 with CUDA support for YOLO model

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# HuggingFace Spaces specific settings
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Create app user (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application files
COPY --chown=user . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/outputs && \
    chown -R user:user /app

# Switch to non-root user
USER user

# Expose port for HuggingFace Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
