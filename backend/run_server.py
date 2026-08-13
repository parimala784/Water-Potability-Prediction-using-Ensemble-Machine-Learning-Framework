"""Run FastAPI server. Usage: python run_server.py"""
import uvicorn
import sys

if __name__ == "__main__":
    # Disable reload on Windows to avoid multiprocessing permission issues
    reload = "--reload" in sys.argv
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=reload)
