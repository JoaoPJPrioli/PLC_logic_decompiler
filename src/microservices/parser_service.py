"""
PLC Parser Microservice
Handles L5X file parsing and tag extraction
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from core.l5x_parser import L5XParser
    from core.processing_pipeline import PLCProcessingPipeline
except ImportError:
    # Mock implementations for standalone deployment
    class L5XParser:
        def parse_file(self, file_path):
            return {"tags": [], "programs": [], "controller": {}}
    
    class PLCProcessingPipeline:
        async def process_file(self, file_path):
            return {"success": True, "data": {}}

app = FastAPI(title="PLC Parser Service", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "parser"}

@app.get("/ready") 
async def readiness_check():
    return {"status": "ready", "service": "parser"}

@app.post("/parse")
async def parse_l5x_file(file: UploadFile = File(...)):
    """Parse uploaded L5X file"""
    try:
        parser = L5XParser()
        pipeline = PLCProcessingPipeline()
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process file
        result = await pipeline.process_file(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return {"success": True, "data": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("parser_service:app", host="0.0.0.0", port=8081)
