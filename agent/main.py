#!/usr/bin/env python3
"""
Proof-of-Work Agent - Main Entry Point

An autonomous AI agent for the Colosseum Solana Agent Hackathon that demonstrates
the observe → think → act → verify loop.

This is a long-running daemon (NOT a web server) designed for:
- Render Background Worker (FREE tier)
- 24/7 operation with automatic restarts

Usage:
    python agent/main.py
"""

import asyncio
import os
import signal
import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import config
from agent.logger import get_logger
from agent.loop import forever, AgentLoop


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    PROOF-OF-WORK AGENT                        ║
║              Colosseum Solana Agent Hackathon                 ║
╠═══════════════════════════════════════════════════════════════╣
║  observe → think → act → verify                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def validate_config():
    """Validate configuration and print status."""
    log = get_logger("main")
    log.info("Validating configuration...")
    
    if hasattr(config, 'validate'):
        errors = config.validate()
        if errors:
            log.error("Configuration errors found:")
            for error in errors:
                log.error(f"  - {error}")
            log.warn("Some features may not work without proper configuration")
    
    # Print config summary (without sensitive values)
    log.info("Configuration loaded:")
    if hasattr(config, 'colosseum'):
        log.info(f"  Colosseum API: {config.colosseum.base_url}")
        log.info(f"  Solana RPC: {config.solana.rpc_url}")
        pid = config.solana.program_id
        log.info(f"  Program ID: {pid[:16]}..." if pid else "  Program ID: NOT SET")
        log.info(f"  Loop interval: {config.agent.loop_interval}s")
        log.info(f"  Log level: {config.agent.log_level}")
    else:
        log.info(f"  Solana RPC: {config.solana_rpc}")
        pid = config.program_id
        log.info(f"  Program ID: {pid[:16]}..." if pid else "  Program ID: NOT SET")
    
    return True


async def main():
    """Main entry point."""
    print_banner()
    
    log = get_logger("main")
    
    # Validate configuration
    validate_config()
    
    # Setup signal handlers for graceful shutdown
    agent = AgentLoop()
    
    def signal_handler(signum, frame):
        log.info(f"Received signal {signum}, initiating shutdown...")
        agent.stop()
    
    # Register signal handlers
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    else:
        signal.signal(signal.SIGINT, signal_handler)
        try:
            signal.signal(signal.SIGBREAK, signal_handler)
        except AttributeError:
            pass
    
    log.info("Starting Proof-of-Work Agent...")
    log.info(f"PID: {os.getpid()}")
    log.info(f"Python: {sys.version}")
    
    try:
        # Run the agent loop forever
        await forever()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        log.info("Agent shutdown complete")


if __name__ == "__main__":
    import os
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).resolve().parent.parent
    os.chdir(script_dir)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested, exiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
