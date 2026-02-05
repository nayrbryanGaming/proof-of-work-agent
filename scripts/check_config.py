#!/usr/bin/env python3
"""
Quick test to verify agent configuration is working.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Load .env
base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")

print("=" * 60)
print("  POW AGENT - CONFIGURATION CHECK")
print("=" * 60)

# Check each required variable
checks = {
    "COLOSSEUM_API_KEY": os.getenv("COLOSSEUM_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "AGENTWALLET_SESSION": os.getenv("AGENTWALLET_SESSION"),
    "PROGRAM_ID": os.getenv("PROGRAM_ID"),
    "SOLANA_RPC": os.getenv("SOLANA_RPC"),
}

all_ok = True
for key, value in checks.items():
    if value:
        # Mask sensitive values
        if "KEY" in key or "SESSION" in key:
            display = value[:20] + "..." if len(value) > 20 else value
        else:
            display = value
        print(f"✅ {key}: {display}")
    else:
        print(f"❌ {key}: NOT SET")
        all_ok = False

print("\n" + "=" * 60)

if all_ok:
    print("✅ All configuration OK! Ready to run agent.")
else:
    print("⚠️  Some configuration missing. Fix before running agent.")
    
# Test Colosseum API
print("\n🔄 Testing Colosseum API...")
import requests

api_key = os.getenv("COLOSSEUM_API_KEY")
if api_key:
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.get(
            "https://agents.colosseum.com/api/agents/status",
            headers=headers,
            timeout=10
        )
        print(f"   Status code: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ API working! Response: {json.dumps(data)[:100]}...")
        else:
            print(f"   ⚠️  API returned: {resp.text[:200]}")
    except Exception as e:
        print(f"   ❌ API error: {e}")

# Test OpenAI API
print("\n🔄 Testing OpenAI API...")
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    try:
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Say 'Agent ready!' in 3 words"}],
                "max_tokens": 10
            },
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
            print(f"   ✅ OpenAI working! Response: {reply}")
        else:
            print(f"   ⚠️  OpenAI error: {resp.status_code} - {resp.text[:100]}")
    except Exception as e:
        print(f"   ❌ OpenAI error: {e}")
else:
    print("   ⚠️  OPENAI_API_KEY not set - agent will use fallback responses")

print("\n" + "=" * 60)
print("  CONFIGURATION CHECK COMPLETE")
print("=" * 60)
