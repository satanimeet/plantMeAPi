#!/usr/bin/env python3
"""
Startup script for Plant Disease API
Works for both local development and Docker deployment
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Docker/Render sets this)
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print("🌱 Starting Plant Disease API...")
    print(f"📍 Host: {host}")
    print(f"📍 Port: {port}")
    print("📚 API documentation will be available at /docs")
    print("🔍 Health check at /health")
    print("🚀 Ready to serve requests!")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
