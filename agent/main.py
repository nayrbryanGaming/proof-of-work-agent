#!/usr/bin/env python3
"""
Proof-of-Work Agent - Main Entry Point (v2.1 Production)

An autonomous AI agent for the Colosseum Solana Agent Hackathon that demonstrates
the observe → think → act → verify loop.

This is a long-running daemon (NOT a web server) designed for:
- Render Background Worker (FREE tier)
- 24/7 operation with automatic restarts
- Self-healing with watchdog
- Comprehensive telemetry and observability

Features:
- Circuit breaker for fault tolerance
- Rate limiting for API protection
- Cryptographic signing for proof verification
- Automatic backup and recovery
- Distributed tracing support

Usage:
    python agent/main.py
"""

import asyncio
import atexit
import os
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import config
from agent.logger import get_logger

# Import all modules for comprehensive functionality
try:
    from agent.loop import forever, AgentLoop
    from agent.shutdown import ShutdownManager, get_shutdown_manager
    from agent.watchdog import Watchdog, RecoveryAction
    from agent.backup import get_backup_manager, quick_snapshot
    from agent.telemetry import get_telemetry, TelemetryManager
    from agent.crypto import get_signer, get_wallet_address
    from agent.errors import get_error_registry
    from agent.state import StateManager
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

VERSION = "2.2.0"
BUILD_DATE = "2026-02-05"


def print_banner():
    """Print startup banner."""
    banner = f"""
╔═══════════════════════════════════════════════════════════════╗
║                    PROOF-OF-WORK AGENT                        ║
║              Colosseum Solana Agent Hackathon                 ║
╠═══════════════════════════════════════════════════════════════╣
║  Version: {VERSION:<20}  Build: {BUILD_DATE:<12} ║
║  observe → think → act → verify                               ║
╠═══════════════════════════════════════════════════════════════╣
║  Features:                                                    ║
║    ✓ Circuit Breaker    ✓ Rate Limiting                      ║
║    ✓ Telemetry          ✓ Crypto Signing                     ║
║    ✓ Auto Backup        ✓ Self-Healing                       ║
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


def initialize_subsystems(log):
    """Initialize all agent subsystems."""
    initialized = []
    
    # Initialize telemetry
    try:
        telemetry = get_telemetry()
        telemetry.configure_output(Path("data/telemetry"))
        initialized.append("telemetry")
        log.info("✓ Telemetry system initialized")
    except Exception as e:
        log.warn(f"✗ Telemetry init failed: {e}")
    
    # Initialize crypto signer
    try:
        signer = get_signer()
        wallet = get_wallet_address()
        initialized.append("crypto")
        log.info(f"✓ Crypto signer initialized: {wallet[:16]}...")
    except Exception as e:
        log.warn(f"✗ Crypto signer init failed: {e}")
    
    # Initialize backup system
    try:
        backup_mgr = get_backup_manager()
        initialized.append("backup")
        log.info("✓ Backup system initialized")
    except Exception as e:
        log.warn(f"✗ Backup system init failed: {e}")
    
    # Create startup snapshot
    try:
        snapshot_name = quick_snapshot("startup")
        log.info(f"✓ Startup snapshot created: {snapshot_name}")
    except Exception as e:
        log.warn(f"✗ Startup snapshot failed: {e}")
    
    return initialized


def setup_shutdown_handlers(agent, log):
    """Setup graceful shutdown handlers."""
    shutdown_manager = get_shutdown_manager()
    
    # Register shutdown hooks
    @shutdown_manager.register(priority=100, timeout=5.0)
    def save_state():
        log.info("Saving agent state...")
        try:
            quick_snapshot("shutdown")
            log.info("State saved successfully")
        except Exception as e:
            log.error(f"Failed to save state: {e}")
    
    @shutdown_manager.register(priority=90, timeout=3.0)
    def stop_agent():
        log.info("Stopping agent loop...")
        agent.stop()
    
    @shutdown_manager.register(priority=80, timeout=5.0)
    def stop_telemetry():
        log.info("Stopping telemetry...")
        try:
            get_telemetry().stop()
        except:
            pass
    
    @shutdown_manager.register(priority=10, timeout=2.0)
    def final_log():
        log.info("Shutdown complete. Goodbye!")
    
    # Signal handlers
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        log.info(f"Received {sig_name}, initiating graceful shutdown...")
        asyncio.create_task(shutdown_manager.shutdown())
    
    # Register signal handlers
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)
    else:
        signal.signal(signal.SIGINT, signal_handler)
        try:
            signal.signal(signal.SIGBREAK, signal_handler)
        except AttributeError:
            pass
    
    # Also register with atexit
    atexit.register(shutdown_manager.sync_shutdown)
    
    return shutdown_manager


async def run_with_watchdog(agent, log):
    """Run agent with watchdog supervision."""
    watchdog = Watchdog()
    
    # Add health checks
    def check_agent_alive():
        return agent.running if hasattr(agent, 'running') else True
    
    def check_memory():
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.percent < 90
        except Exception:
            return True
    
    watchdog.register_check("agent_alive", check_agent_alive, interval=30.0)
    watchdog.register_check("memory_ok", check_memory, interval=60.0)
    
    # Add recovery action
    def restart_agent():
        log.warn("Watchdog triggering agent restart...")
        if hasattr(agent, 'stop'):
            agent.stop()
        asyncio.create_task(forever())
    
    watchdog.register_recovery_handler(RecoveryAction.FULL_RESTART, restart_agent)
    
    # Start watchdog
    await watchdog.start()
    
    try:
        # Start telemetry collection
        try:
            await get_telemetry().start()
        except Exception as e:
            log.warn(f"Telemetry start failed: {e}")
        
        # Run the agent loop
        await forever()
    finally:
        await watchdog.stop()


async def main():
    """Main entry point."""
    print_banner()
    
    log = get_logger("main")
    log.info(f"POW Agent v{VERSION} starting...")
    log.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    # Validate configuration
    validate_config()
    
    # Initialize subsystems
    initialized = initialize_subsystems(log)
    log.info(f"Initialized {len(initialized)} subsystems: {', '.join(initialized)}")
    
    # Setup agent
    agent = AgentLoop()
    
    # Setup shutdown handlers
    shutdown_manager = setup_shutdown_handlers(agent, log)
    
    log.info("=" * 60)
    log.info("Starting Proof-of-Work Agent...")
    log.info(f"PID: {os.getpid()}")
    log.info(f"Python: {sys.version.split()[0]}")
    log.info(f"Platform: {sys.platform}")
    log.info("=" * 60)
    
    try:
        # Run with watchdog supervision
        await run_with_watchdog(agent, log)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")
    except asyncio.CancelledError:
        log.info("Agent task cancelled")
    except Exception as e:
        log.error(f"Fatal error: {e}")
        traceback.print_exc()
        
        # Record error for post-mortem
        try:
            get_error_registry().record(e)
        except:
            pass
        
        sys.exit(1)
    finally:
        log.info("Agent main loop exited")


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
