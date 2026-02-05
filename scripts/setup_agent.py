#!/usr/bin/env python3
"""
Setup script for POW Agent - Generates all required configurations.
Run this ONCE to set up your agent.
"""

import os
import sys
import json
import secrets
import hashlib
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# Pure Python Ed25519 Implementation (from solana/client.py)
# ============================================================

b = 256
q = 2**255 - 19
l = 2**252 + 27742317777372353535851937790883648493

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
    for b in data:
        if b == 0:
            pad += 1
        else:
            break
    return ("1" * pad) + (res or "1")

# ============================================================
# Keypair Generation
# ============================================================

def generate_keypair():
    """Generate a new Solana keypair."""
    # Generate 32 random bytes as seed
    seed = secrets.token_bytes(32)
    
    # Derive public key
    public_key = ed25519_publickey(seed)
    
    # Full secret is seed + public_key (64 bytes, Solana format)
    full_secret = seed + public_key
    
    return {
        "seed": seed,
        "public_key": public_key,
        "full_secret": full_secret,
        "public_key_b58": b58encode(public_key),
        "secret_array": list(full_secret),
    }

# ============================================================
# Colosseum API Registration
# ============================================================

def register_agent(agent_name: str) -> dict:
    """Register agent with Colosseum API."""
    import requests
    
    url = "https://agents.colosseum.com/api/agents"
    payload = {"name": agent_name}
    
    print(f"\n🔄 Registering agent '{agent_name}'...")
    
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200 or resp.status_code == 201:
            data = resp.json()
            print("✅ Agent registered successfully!")
            return data
        else:
            print(f"❌ Registration failed: {resp.status_code}")
            print(f"   Response: {resp.text}")
            return {}
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return {}

# ============================================================
# Main Setup
# ============================================================

def main():
    print("=" * 60)
    print("  POW AGENT SETUP WIZARD")
    print("=" * 60)
    
    base_dir = Path(__file__).resolve().parent.parent
    env_file = base_dir / ".env"
    keypair_file = base_dir / "data" / "agent_keypair.json"
    
    # Ensure data directory exists
    (base_dir / "data").mkdir(exist_ok=True)
    
    # ============================================================
    # Step 1: Generate Solana Keypair
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Generate Solana Keypair")
    print("=" * 60)
    
    if keypair_file.exists():
        print(f"⚠️  Keypair already exists at: {keypair_file}")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            with open(keypair_file, 'r') as f:
                keypair_data = json.load(f)
            print("✅ Using existing keypair")
        else:
            keypair = generate_keypair()
            keypair_data = {
                "public_key": keypair["public_key_b58"],
                "secret_array": keypair["secret_array"],
            }
            with open(keypair_file, 'w') as f:
                json.dump(keypair_data, f, indent=2)
            print(f"✅ New keypair generated!")
    else:
        keypair = generate_keypair()
        keypair_data = {
            "public_key": keypair["public_key_b58"],
            "secret_array": keypair["secret_array"],
        }
        with open(keypair_file, 'w') as f:
            json.dump(keypair_data, f, indent=2)
        print(f"✅ Keypair generated!")
    
    print(f"\n📍 Public Key (Wallet Address):")
    print(f"   {keypair_data['public_key']}")
    print(f"\n💾 Keypair saved to: {keypair_file}")
    
    # ============================================================
    # Step 2: Register with Colosseum
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Register Agent with Colosseum")
    print("=" * 60)
    
    agent_name = input("\nEnter your agent name (e.g., pow-agent-nayrbryan): ").strip()
    if not agent_name:
        agent_name = f"pow-agent-{secrets.token_hex(4)}"
        print(f"Using random name: {agent_name}")
    
    registration = register_agent(agent_name)
    
    colosseum_api_key = ""
    claim_code = ""
    
    if registration:
        colosseum_api_key = registration.get("apiKey", "")
        claim_code = registration.get("claimCode", "")
        
        if colosseum_api_key:
            print(f"\n🔑 COLOSSEUM_API_KEY: {colosseum_api_key}")
            print(f"🎫 CLAIM_CODE: {claim_code}")
            print("\n⚠️  SAVE THESE! The API key is shown ONCE and cannot be recovered!")
    
    # ============================================================
    # Step 3: Get OpenAI API Key
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: OpenAI API Key")
    print("=" * 60)
    
    print("\nGet your API key from: https://platform.openai.com/api-keys")
    openai_api_key = input("Enter your OpenAI API Key (or press Enter to skip): ").strip()
    
    # ============================================================
    # Step 4: Program ID
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Anchor Program ID")
    print("=" * 60)
    
    print("\nYou need to deploy the Anchor program first:")
    print("  cd solana/program")
    print("  anchor build")
    print("  anchor deploy")
    print("  Copy the Program Id from output")
    
    program_id = input("\nEnter Program ID (or press Enter to skip for now): ").strip()
    
    # ============================================================
    # Step 5: Create .env file
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: Create .env File")
    print("=" * 60)
    
    secret_array_str = json.dumps(keypair_data["secret_array"])
    
    env_content = f"""# ============================================================
# POW AGENT - Environment Configuration
# Generated by setup_agent.py
# ============================================================

# Colosseum API
COLOSSEUM_API_KEY={colosseum_api_key}
COLOSSEUM_BASE_URL=https://agents.colosseum.com/api

# OpenAI
OPENAI_API_KEY={openai_api_key}
OPENAI_MODEL=gpt-4

# Solana Wallet (AgentWallet)
AGENTWALLET_SESSION={secret_array_str}

# Solana Network
SOLANA_RPC=https://api.devnet.solana.com
PROGRAM_ID={program_id}
BOUNTY_ID=1

# Agent Settings
HEARTBEAT_URL=https://agents.colosseum.com/heartbeat.md
LOOP_INTERVAL_SECONDS=1800
LOG_LEVEL=INFO
ENVIRONMENT=development
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ .env file created at: {env_file}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    
    print(f"""
📋 YOUR CONFIGURATION:

🔑 COLOSSEUM_API_KEY: {"✅ Set" if colosseum_api_key else "❌ Not set"}
🧠 OPENAI_API_KEY: {"✅ Set" if openai_api_key else "❌ Not set"}
💰 AGENTWALLET_SESSION: ✅ Generated
🧾 PROGRAM_ID: {"✅ Set" if program_id else "❌ Not set (deploy Anchor first)"}

📍 Your Wallet Address: {keypair_data['public_key']}

⚠️  IMPORTANT:
1. Fund your wallet with devnet SOL:
   Visit: https://faucet.solana.com
   Paste: {keypair_data['public_key']}

2. Deploy Anchor program (if not done):
   cd solana/program
   anchor build
   anchor deploy

3. Update PROGRAM_ID in .env after deploy

4. For Render deployment, copy these values to Environment Variables:
   - COLOSSEUM_API_KEY
   - OPENAI_API_KEY
   - AGENTWALLET_SESSION
   - PROGRAM_ID
   - SOLANA_RPC

5. Run locally to test:
   python agent/main.py
""")
    
    if claim_code:
        print(f"🎫 CLAIM CODE (give to human for prize): {claim_code}")
    
    print("\n✅ Ready to run your autonomous agent!")


if __name__ == "__main__":
    main()
