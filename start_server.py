#!/usr/bin/env python3
"""
Simple startup script for the breast cancer detection API
"""

import uvicorn
from main import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Breast Cancer Detection API Server")
    print("="*60)
    print("Starting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )