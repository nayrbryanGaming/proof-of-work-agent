"""
Run the API server for the POW Agent Dashboard.
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

from api.server import app


def main():
    """Run the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║               POW AGENT API SERVER                            ║
╠═══════════════════════════════════════════════════════════════╣
║  Host: {host}:{port}                                           
║  Docs: http://{host}:{port}/api/docs                           
║  Health: http://{host}:{port}/health                           
╚═══════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
