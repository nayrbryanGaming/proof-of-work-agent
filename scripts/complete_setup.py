#!/usr/bin/env python3
"""
Complete Setup Script for POW Agent
Handles:
1. Solana keypair generation/loading
2. Agent registration with Colosseum
3. Environment configuration
4. Validation of all components
5. Faucet airdrop (optional)
"""

import os
import sys
import json
import hashlib
import time
import requests
from pathlib import Path
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===========================================================
# Pure Python Ed25519 Implementation (No Solana CLI needed!)
# ===========================================================

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ALPHABET_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    res = ""
    while n > 0:
        n, rem = divmod(n, 58)
        res = ALPHABET[rem] + res
    pad = sum(1 for b in data if b == 0)
    return ("1" * pad) + (res or "")


def b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        if ch not in ALPHABET_INDEX:
            raise ValueError("Invalid base58")
        n = n * 58 + ALPHABET_INDEX[ch]
    full = n.to_bytes((n.bit_length() + 7) // 8, "big") if n > 0 else b""
    pad = sum(1 for ch in s if ch == "1")
    return b"\x00" * pad + full


# Ed25519 constants
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

def edwards(P, Q):
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * inv(1 + d * x1 * x2 * y1 * y2) % q
    y3 = (y1 * y2 + x1 * x2) * inv(1 - d * x1 * x2 * y1 * y2) % q
    return (x3, y3)

def scalarmult(P, e):
    if e == 0:
        return (0, 1)
    Q = (0, 1)
    N = P
    while e > 0:
        if e & 1:
            Q = edwards(Q, N)
        N = edwards(N, N)
        e >>= 1
    return Q

def encodepoint(P):
    x, y = P
    bits = y | ((x & 1) << 255)
    return bits.to_bytes(32, "little")

By = (4 * inv(5)) % q
Bx = xrecover(By)
B = (Bx, By)

def secret_expand(seed):
    h = H(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]

def ed25519_publickey(seed):
    a, _ = secret_expand(seed)
    A = scalarmult(B, a)
    return encodepoint(A)


class Keypair:
    """Solana-compatible keypair using pure Python Ed25519."""
    
    def __init__(self, secret: bytes):
        if len(secret) == 64:
            self.seed = secret[:32]
        elif len(secret) == 32:
            self.seed = secret
        else:
            raise ValueError("Invalid secret length")
        self.public_key = ed25519_publickey(self.seed)
    
    @classmethod
    def generate(cls) -> "Keypair":
        seed = os.urandom(32)
        return cls(seed)
    
    @classmethod
    def from_file(cls, path: str) -> "Keypair":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(bytes(data))
    
    def to_bytes(self) -> bytes:
        return self.seed + self.public_key
    
    def to_json(self) -> list:
        return list(self.seed + self.public_key)
    
    def pubkey_base58(self) -> str:
        return b58encode(self.public_key)
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)


# ===========================================================
# Setup Functions
# ===========================================================

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          POW AGENT - COMPLETE SETUP WIZARD                   ║
║                  Solana Devnet Ready                         ║
╚══════════════════════════════════════════════════════════════╝
""")


def get_or_create_keypair() -> Keypair:
    """Get existing keypair or create a new one."""
    
    # Priority order for keypair sources
    paths_to_check = [
        Path.home() / ".config" / "solana" / "id.json",  # Standard Solana CLI location
        Path(__file__).parent.parent / "data" / "keypair.json",  # Project local
        Path(__file__).parent.parent / ".keypair.json",  # Project root
    ]
    
    # Check existing env
    env_session = os.getenv("AGENTWALLET_SESSION", "").strip()
    if env_session and env_session.startswith("["):
        try:
            data = json.loads(env_session)
            kp = Keypair(bytes(data))
            print(f"✅ Using keypair from AGENTWALLET_SESSION")
            print(f"   Address: {kp.pubkey_base58()}")
            return kp
        except:
            pass
    
    # Check file paths
    for path in paths_to_check:
        if path.exists():
            try:
                kp = Keypair.from_file(str(path))
                print(f"✅ Loaded keypair from {path}")
                print(f"   Address: {kp.pubkey_base58()}")
                return kp
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
    
    # Generate new keypair
    print("🔑 Generating new Solana keypair...")
    kp = Keypair.generate()
    
    # Save to project directory
    save_path = Path(__file__).parent.parent / "data" / "keypair.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    kp.save(str(save_path))
    
    print(f"✅ Generated new keypair")
    print(f"   Address: {kp.pubkey_base58()}")
    print(f"   Saved to: {save_path}")
    
    return kp


def check_colosseum_registration(api_key: str) -> dict | None:
    """Check if agent is registered with Colosseum."""
    try:
        resp = requests.get(
            "https://agents.colosseum.com/api/agents/status",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


def register_agent(wallet_address: str) -> dict:
    """Register agent with Colosseum."""
    print("\n📝 Registering agent with Colosseum...")
    
    # Generate unique agent name
    suffix = hashlib.sha256(wallet_address.encode()).hexdigest()[:8]
    agent_name = f"pow-agent-{os.getenv('USERNAME', 'agent')}-{suffix}"
    
    payload = {
        "name": agent_name,
        "walletAddress": wallet_address,
        "description": "Autonomous proof-of-work agent for the Colosseum Solana Agent Hackathon"
    }
    
    try:
        resp = requests.post(
            "https://agents.colosseum.com/api/agents",
            json=payload,
            timeout=15
        )
        
        if resp.status_code in (200, 201):
            data = resp.json()
            print(f"✅ Agent registered successfully!")
            return data
        else:
            print(f"⚠️  Registration response: {resp.status_code}")
            print(f"   Body: {resp.text[:200]}")
            return {"name": agent_name}
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return {"name": agent_name}


def request_airdrop(wallet_address: str, amount: float = 2.0) -> bool:
    """Request SOL airdrop from devnet faucet."""
    print(f"\n🚰 Requesting {amount} SOL airdrop...")
    
    try:
        # Using Solana devnet RPC
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "requestAirdrop",
            "params": [wallet_address, int(amount * 1_000_000_000)]
        }
        
        resp = requests.post(
            "https://api.devnet.solana.com",
            json=payload,
            timeout=30
        )
        
        data = resp.json()
        if "result" in data:
            print(f"✅ Airdrop requested! TX: {data['result'][:20]}...")
            print(f"   Waiting for confirmation...")
            time.sleep(5)  # Wait for confirmation
            return True
        else:
            print(f"⚠️  Airdrop response: {data}")
            return False
    except Exception as e:
        print(f"❌ Airdrop failed: {e}")
        print(f"   Manual faucet: https://faucet.solana.com")
        return False


def check_balance(wallet_address: str) -> float:
    """Check SOL balance."""
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [wallet_address]
        }
        resp = requests.post(
            "https://api.devnet.solana.com",
            json=payload,
            timeout=10
        )
        data = resp.json()
        if "result" in data:
            lamports = data["result"]["value"]
            sol = lamports / 1_000_000_000
            return sol
    except:
        pass
    return 0.0


def update_env_file(keypair: Keypair, registration: dict, env_path: Path):
    """Update .env file with credentials."""
    
    # Read existing .env
    existing = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                existing[key.strip()] = value.strip()
    
    # Update values
    updates = {
        "AGENTWALLET_SESSION": json.dumps(keypair.to_json()),
        "WALLET_ADDRESS": keypair.pubkey_base58(),
    }
    
    # Add registration data if available
    if registration.get("apiKey"):
        updates["COLOSSEUM_API_KEY"] = registration["apiKey"]
    if registration.get("claimCode"):
        updates["CLAIM_CODE"] = registration["claimCode"]
    if registration.get("name"):
        updates["AGENT_NAME"] = registration["name"]
    
    existing.update(updates)
    
    # Write .env file
    lines = [
        "# ============================================================",
        "# POW AGENT - Environment Configuration",
        f"# Generated: {datetime.now().isoformat()}",
        "# ============================================================",
        "",
        "# Colosseum API",
        f"COLOSSEUM_API_KEY={existing.get('COLOSSEUM_API_KEY', '')}",
        f"COLOSSEUM_BASE_URL={existing.get('COLOSSEUM_BASE_URL', 'https://agents.colosseum.com/api')}",
        "",
        "# OpenAI - GET FROM: https://platform.openai.com/api-keys",
        f"OPENAI_API_KEY={existing.get('OPENAI_API_KEY', '')}",
        f"OPENAI_MODEL={existing.get('OPENAI_MODEL', 'gpt-4')}",
        "",
        "# Solana Wallet",
        f"AGENTWALLET_SESSION={existing.get('AGENTWALLET_SESSION', '')}",
        f"WALLET_ADDRESS={existing.get('WALLET_ADDRESS', '')}",
        "",
        "# Solana Network",
        f"SOLANA_RPC={existing.get('SOLANA_RPC', 'https://api.devnet.solana.com')}",
        f"PROGRAM_ID={existing.get('PROGRAM_ID', '')}",
        f"BOUNTY_ID={existing.get('BOUNTY_ID', '1')}",
        "",
        "# Agent Settings",
        f"HEARTBEAT_URL={existing.get('HEARTBEAT_URL', 'https://agents.colosseum.com/heartbeat.md')}",
        f"LOOP_INTERVAL_SECONDS={existing.get('LOOP_INTERVAL_SECONDS', '1800')}",
        f"LOG_LEVEL={existing.get('LOG_LEVEL', 'INFO')}",
        f"ENVIRONMENT={existing.get('ENVIRONMENT', 'development')}",
        "",
        "# Registration Info",
        f"AGENT_NAME={existing.get('AGENT_NAME', '')}",
        f"CLAIM_CODE={existing.get('CLAIM_CODE', '')}",
        ""
    ]
    
    env_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Updated {env_path}")


def validate_setup() -> dict:
    """Validate all setup components."""
    print("\n🔍 Validating setup...")
    
    checks = {}
    
    # Check required files
    base = Path(__file__).parent.parent
    
    checks["requirements.txt"] = (base / "requirements.txt").exists()
    checks["start.sh"] = (base / "start.sh").exists()
    checks["Procfile"] = (base / "Procfile").exists()
    checks["agent/main.py"] = (base / "agent" / "main.py").exists()
    
    # Check environment
    from dotenv import load_dotenv
    load_dotenv(base / ".env")
    
    checks["COLOSSEUM_API_KEY"] = bool(os.getenv("COLOSSEUM_API_KEY"))
    checks["AGENTWALLET_SESSION"] = bool(os.getenv("AGENTWALLET_SESSION"))
    checks["OPENAI_API_KEY"] = bool(os.getenv("OPENAI_API_KEY"))
    
    # Print results
    for name, ok in checks.items():
        status = "✅" if ok else "❌"
        print(f"   {status} {name}")
    
    return checks


def main():
    """Run complete setup."""
    print_banner()
    
    base_path = Path(__file__).parent.parent
    env_path = base_path / ".env"
    
    # Step 1: Get or create keypair
    print("=" * 60)
    print("STEP 1: Solana Wallet Setup")
    print("=" * 60)
    keypair = get_or_create_keypair()
    wallet_address = keypair.pubkey_base58()
    
    # Step 2: Check balance and request airdrop if needed
    print("\n" + "=" * 60)
    print("STEP 2: Check Balance & Airdrop")
    print("=" * 60)
    balance = check_balance(wallet_address)
    print(f"💰 Current balance: {balance:.4f} SOL")
    
    if balance < 0.5:
        request_airdrop(wallet_address)
        time.sleep(3)
        balance = check_balance(wallet_address)
        print(f"💰 New balance: {balance:.4f} SOL")
    
    # Step 3: Check/Register with Colosseum
    print("\n" + "=" * 60)
    print("STEP 3: Colosseum Registration")
    print("=" * 60)
    
    # Load existing API key
    from dotenv import load_dotenv
    load_dotenv(env_path)
    existing_key = os.getenv("COLOSSEUM_API_KEY", "").strip()
    
    registration = {}
    if existing_key:
        status = check_colosseum_registration(existing_key)
        if status:
            print(f"✅ Already registered with Colosseum")
            registration = {"apiKey": existing_key}
        else:
            registration = register_agent(wallet_address)
    else:
        registration = register_agent(wallet_address)
    
    # Step 4: Update .env file
    print("\n" + "=" * 60)
    print("STEP 4: Update Configuration")
    print("=" * 60)
    update_env_file(keypair, registration, env_path)
    
    # Step 5: Validate setup
    print("\n" + "=" * 60)
    print("STEP 5: Validation")
    print("=" * 60)
    checks = validate_setup()
    
    # Final summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f"""
📍 Wallet Address: {wallet_address}
💰 Balance: {balance:.4f} SOL
🔗 Solana Explorer: https://explorer.solana.com/address/{wallet_address}?cluster=devnet

Next Steps:
1. Add OPENAI_API_KEY to .env (from https://platform.openai.com/api-keys)
2. Deploy Anchor program and set PROGRAM_ID (optional)
3. Run: python agent/main.py
4. Deploy to Render as Background Worker

Manual Faucet (if airdrop failed):
   https://faucet.solana.com
   Address: {wallet_address}
""")
    
    # Check for missing items
    if not checks.get("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set - agent will use fallback responses")
    
    if not os.getenv("PROGRAM_ID"):
        print("ℹ️  INFO: PROGRAM_ID not set - running in TEST MODE (no on-chain transactions)")


if __name__ == "__main__":
    main()
