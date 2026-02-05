#!/usr/bin/env python3
"""
Run a single test cycle of the POW Agent.
Useful for testing without waiting 30 minutes.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Override loop interval for testing
os.environ["LOOP_INTERVAL_SECONDS"] = "10"

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agent.logger import get_logger
from agent.loop import run_cycle, get_solana_client
from colosseum.api import ColosseumAPI


async def main():
    log = get_logger("test_cycle")
    
    print("=" * 60)
    print("  POW AGENT - SINGLE CYCLE TEST")
    print("=" * 60)
    
    # Initialize clients
    api = ColosseumAPI()
    solana = get_solana_client()
    
    if solana is None:
        print("\n⚠️  Running in TEST MODE (no Solana on-chain)")
    else:
        print("\n✅ Solana client initialized")
    
    print("\n🔄 Running single cycle...")
    print("-" * 60)
    
    try:
        result = await run_cycle(api, solana, cycle_num=1)
        
        print("-" * 60)
        print("\n📊 CYCLE RESULT:")
        print(f"   Heartbeat synced: {result.heartbeat_synced}")
        print(f"   Status checked: {result.status_checked}")
        print(f"   Forum engaged: {result.forum_engaged}")
        print(f"   Task solved: {result.task_solved}")
        print(f"   Task hash: {result.task_hash[:32] if result.task_hash else 'None'}...")
        print(f"   Solana TX: {result.solana_tx or 'None'}")
        print(f"   Project updated: {result.project_updated}")
        print(f"   Duration: {result.duration:.2f}s")
        
        if result.errors:
            print(f"\n⚠️  ERRORS ({len(result.errors)}):")
            for err in result.errors:
                print(f"   - {err[:80]}")
        else:
            print("\n✅ No errors!")
            
    except Exception as e:
        print(f"\n❌ Cycle failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
