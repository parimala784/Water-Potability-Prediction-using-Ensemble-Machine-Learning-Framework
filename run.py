"""
Run frontend and backend together from a single script.
Usage: python run.py

Press Ctrl+C to stop both.
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def main():
    if not BACKEND_DIR.exists():
        print("Error: backend folder not found")
        sys.exit(1)
    if not FRONTEND_DIR.exists():
        print("Error: frontend folder not found")
        sys.exit(1)

    processes = []

    def cleanup():
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        sys.exit(0)

    def handler(signum, frame):
        print("\nStopping frontend and backend...")
        cleanup()

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)

    # Prefer backend venv Python if it exists
    if sys.platform == "win32":
        venv_python = BACKEND_DIR / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = BACKEND_DIR / "venv" / "bin" / "python"
    backend_python = str(venv_python) if venv_python.exists() else sys.executable

    print("Starting backend (http://localhost:8000)...")
    backend = subprocess.Popen(
        [backend_python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=BACKEND_DIR,
        shell=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    processes.append(backend)

    print("Starting frontend (http://localhost:5173)...")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True,  # needed on Windows for npm
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    processes.append(frontend)

    print("\nBackend:  http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("Press Ctrl+C to stop both.\n")

    # Wait for either process to exit
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
