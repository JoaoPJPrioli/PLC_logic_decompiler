"""
API Gateway for PLC Logic Decompiler Microservices
Centralized entry point for all microservice communication
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn

app = FastAPI(
    title="PLC Decompiler API Gateway",
    description="Centralized API Gateway for PLC Logic Decompiler microservices",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service registry
SERVICE_REGISTRY = {
    "parser": "http://parser-service:8081",
    "analysis": "http://analysis-service:8082", 
    "ai": "http://ai-service:8083",
    "visualization": "http://visualization-service:8084"
}

security = HTTPBearer()

class ServiceRegistry:
    """Service discovery and registry"""
    
    def __init__(self):
        self.services = SERVICE_REGISTRY
        self.health_status = {}
    
    async def get_service_url(self, service_name: str) -> str:
        """Get service URL by name"""
        if service_name not in self.services:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        return self.services[service_name]
    
    async def health_check(self, service_name: str) -> bool:
        """Check service health"""
        try:
            service_url = await self.get_service_url(service_name)
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

registry = ServiceRegistry()

@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/services/status")
async def services_status():
    """Check status of all services"""
    status = {}
    for service_name in SERVICE_REGISTRY.keys():
        status[service_name] = await registry.health_check(service_name)
    return {"services": status, "timestamp": datetime.now().isoformat()}

@app.post("/api/parse")
async def parse_l5x_file(request: Request):
    """Route to parser service"""
    service_url = await registry.get_service_url("parser")
    
    async with httpx.AsyncClient() as client:
        # Forward request to parser service
        response = await client.post(
            f"{service_url}/parse",
            content=await request.body(),
            headers=dict(request.headers),
            timeout=300.0
        )
        return response.json()

@app.post("/api/analyze")
async def analyze_plc_data(request: Request):
    """Route to analysis service"""
    service_url = await registry.get_service_url("analysis")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{service_url}/analyze",
            content=await request.body(),
            headers=dict(request.headers),
            timeout=600.0
        )
        return response.json()

@app.post("/api/generate-code")
async def generate_code(request: Request):
    """Route to AI service for code generation"""
    service_url = await registry.get_service_url("ai")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{service_url}/generate",
            content=await request.body(),
            headers=dict(request.headers),
            timeout=900.0
        )
        return response.json()

@app.post("/api/visualize")
async def create_visualization(request: Request):
    """Route to visualization service"""
    service_url = await registry.get_service_url("visualization")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{service_url}/create",
            content=await request.body(),
            headers=dict(request.headers),
            timeout=300.0
        )
        return response.json()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    logging.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
