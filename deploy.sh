#!/bin/bash
# Deploy PLC Decompiler to Kubernetes

ENVIRONMENT=${1:-development}
NAMESPACE="plc-decompiler-${ENVIRONMENT}"

echo "🚀 Deploying PLC Decompiler to ${ENVIRONMENT}..."

# Create namespace
kubectl apply -f k8s/${ENVIRONMENT}/namespace.yaml

# Deploy services
kubectl apply -f k8s/${ENVIRONMENT}/ -n ${NAMESPACE}

# Wait for deployments
kubectl wait --for=condition=ready pod -l app=api-gateway -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app=parser-service -n ${NAMESPACE} --timeout=300s

echo "✅ Deployment complete!"
echo "🌐 Access the application at: https://plc-decompiler-${ENVIRONMENT}.example.com"
