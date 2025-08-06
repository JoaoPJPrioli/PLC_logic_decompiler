#!/bin/bash
# Build all Docker images for PLC Decompiler microservices

echo "🐳 Building PLC Decompiler Docker Images..."

# Build services
docker build -f docker/api-gateway.Dockerfile -t plc-decompiler/api-gateway:latest .
docker build -f docker/parser-service.Dockerfile -t plc-decompiler/parser-service:latest .
docker build -f docker/analysis-service.Dockerfile -t plc-decompiler/analysis-service:latest .
docker build -f docker/ai-service.Dockerfile -t plc-decompiler/ai-service:latest .
docker build -f docker/visualization-service.Dockerfile -t plc-decompiler/visualization-service:latest .
docker build -f docker/web-frontend.Dockerfile -t plc-decompiler/web-frontend:latest .

echo "✅ All images built successfully!"
