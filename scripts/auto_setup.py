#!/usr/bin/env python3
"""
Auto-setup script for POW Agent - Generates all required configurations.
Non-interactive version.
"""

import os
import sys
import json
import secrets
import hashlib
from pathlib import Path

print("=" * 60)
print("  POW AGENT AUTO-SETUP")
print("=" * 60)

# ============================================================
# Pure Python Ed25519 Implementation
# ============================================================

b = 256
q = 2**255 - 19
l_param = 2**252 + 27742317777372353535851937790883648493

def H(m: bytes) -> bytes:
    return hashlib.sha512(m).digest()

def inv(x: int) -> int:
    return pow(x, q - 2, q)

d = (-121665 * inv(121666)) % q
I = pow(2, (q - 1) // 4, q)

def xrecover(y: int) -> int:
    xx = (y * y - 1) * inv(d * y * y + 1) % q
    x = pow(xx, (q + 3) // 8, q)
    if (x * x - xx) % q != 0:
        x = (x * I) % q
    if x % 2 != 0:
        x = q - x
    return x

def scalarmult(P, e):
    if e == 0:
        return (0, 1)
    Q = (0, 1)
    N = P
    k = e
    while k > 0:
        if k & 1:
            x1, y1 = Q
            x2, y2 = N
            x3 = (x1 * y2 + x2 * y1) * inv(1 + d * x1 * x2 * y1 * y2) % q
            y3 = (y1 * y2 + x1 * x2) * inv(1 - d * x1 * x2 * y1 * y2) % q
            Q = (x3, y3)
        x1, y1 = N
        x3 = (x1 * y1 + x1 * y1) * inv(1 + d * x1 * x1 * y1 * y1) % q
        y3 = (y1 * y1 + x1 * x1) * inv(1 - d * x1 * x1 * y1 * y1) % q
        N = (x3, y3)
        k >>= 1
    return Q

def encodepoint(P) -> bytes:
    x, y = P
    bits = y | ((x & 1) << 255)
    return bits.to_bytes(32, "little")

By = (4 * inv(5)) % q
Bx = xrecover(By)
B = (Bx, By)

def _secret_expand(seed: bytes):
    h = H(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a

def ed25519_publickey(seed: bytes) -> bytes:
    a = _secret_expand(seed)
    A = scalarmult(B, a)
    return encodepoint(A)

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    res = ""
    while n > 0:
        n, rem = divmod(n, 58)
        res = ALPHABET[rem] + res
    pad = 0
    for byte in data:
        if byte == 0:
            pad += 1
        else:
            break
    return ("1" * pad) + (res or "1")

# ============================================================
# Generate Keypair
# ============================================================

print("\n[1/3] Generating Solana Keypair...")

seed = secrets.token_bytes(32)
public_key = ed25519_publickey(seed)
full_secret = seed + public_key
public_key_b58 = b58encode(public_key)
secret_array = list(full_secret)

print(f"✅ Keypair generated!")
print(f"📍 Public Key (Wallet Address): {public_key_b58}")

# Save keypair
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
data_dir.mkdir(exist_ok=True)

keypair_file = data_dir / "agent_keypair.json"
with open(keypair_file, 'w') as f:
    json.dump({
        "public_key": public_key_b58,
        "secret_array": secret_array,
    }, f, indent=2)
print(f"💾 Saved to: {keypair_file}")

# ============================================================
# Register with Colosseum
# ============================================================

print("\n[2/3] Registering with Colosseum...")

import requests

agent_name = f"pow-agent-nayrbryan-{secrets.token_hex(4)}"
url = "https://agents.colosseum.com/api/agents"

try:
    resp = requests.post(url, json={"name": agent_name}, timeout=30)
    print(f"Response status: {resp.status_code}")
    
    if resp.status_code in (200, 201):
        data = resp.json()
        colosseum_api_key = data.get("apiKey", "")
        claim_code = data.get("claimCode", "")
        
        print(f"✅ Agent registered: {agent_name}")
        print(f"🔑 COLOSSEUM_API_KEY: {colosseum_api_key}")
        print(f"🎫 CLAIM_CODE: {claim_code}")
        print("\n⚠️  SAVE THESE VALUES! API key shown ONCE only!")
    else:
        print(f"❌ Registration failed: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
        colosseum_api_key = ""
        claim_code = ""
except Exception as e:
    print(f"❌ Registration error: {e}")
    colosseum_api_key = ""
    claim_code = ""

# ============================================================
# Create .env file
# ============================================================

print("\n[3/3] Creating .env file...")

secret_array_str = json.dumps(secret_array)

env_content = f"""# ============================================================
# POW AGENT - Environment Configuration
# Auto-generated - {__import__('datetime').datetime.now().isoformat()}
# ============================================================

# Colosseum API
COLOSSEUM_API_KEY={colosseum_api_key}
COLOSSEUM_BASE_URL=https://agents.colosseum.com/api

# OpenAI - GET FROM: https://platform.openai.com/api-keys
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4

# Solana Wallet (AgentWallet)
AGENTWALLET_SESSION={secret_array_str}

# Solana Network
SOLANA_RPC=https://api.devnet.solana.com
PROGRAM_ID=
BOUNTY_ID=1

# Agent Settings
HEARTBEAT_URL=https://agents.colosseum.com/heartbeat.md
LOOP_INTERVAL_SECONDS=1800
LOG_LEVEL=INFO
ENVIRONMENT=development
"""

env_file = base_dir / ".env"
with open(env_file, 'w') as f:
    f.write(env_content)

print(f"✅ .env created at: {env_file}")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("  SETUP COMPLETE!")
print("=" * 60)

print(f"""
📋 YOUR CONFIGURATION:

🔑 COLOSSEUM_API_KEY: {"✅ " + colosseum_api_key[:20] + "..." if colosseum_api_key else "❌ Not set"}
🧠 OPENAI_API_KEY: ❌ Not set (add to .env)
💰 AGENTWALLET_SESSION: ✅ Generated
🧾 PROGRAM_ID: ❌ Not set (deploy Anchor first)

📍 Your Wallet Address: {public_key_b58}

⚠️  NEXT STEPS:

1. Fund your wallet with devnet SOL:
   Visit: https://faucet.solana.com
   Paste your wallet address above

2. Add your OpenAI API key to .env:
   Get from: https://platform.openai.com/api-keys

3. Deploy Anchor program:
   cd solana/program
   anchor build
   anchor deploy
   Copy the Program Id to .env

4. Run agent:
   python agent/main.py
""")

if claim_code:
    print(f"🎫 CLAIM CODE (give to human for prize claiming): {claim_code}")

print("\n✅ Ready!")
