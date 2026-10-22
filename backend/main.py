"""
GridSathi Main Entry Point
Serves the FastAPI application with uvicorn.
Supports being run from the project root or from within the backend directory.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import uvicorn

try:
    from backend.app import app
except ImportError:
    from app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting GridSathi: Energy Forecasting & Smart Demand Scheduling on http://{host}:{port} ...")
    uvicorn.run(app, host=host, port=port)
