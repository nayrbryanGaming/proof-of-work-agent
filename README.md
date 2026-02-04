# Proof-of-Work Agent

![POW Agent](https://img.shields.io/badge/POW-Agent-blue)
![Solana](https://img.shields.io/badge/Solana-Devnet-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An autonomous AI agent for the Colosseum Solana Agent Hackathon that demonstrates the **observe → think → act → verify** loop.

## ⚠️ ARCHITECTURE

**This is a long-running Python daemon, NOT a web server.**

- Runs 24/7 as a background worker
- Designed for **Render Background Worker** (FREE tier)
- NO FastAPI, NO Express, NO serverless functions
- Pure asyncio event loop

## 🚀 Features

- 🔄 **Autonomous Loop** - Polls heartbeat.md every 30 minutes
- 📊 **Decision Engine** - Uses `/agents/status` as decision signal
- 💬 **Forum Engagement** - Reads, votes, and comments meaningfully on forum posts
- 📝 **Project Management** - Creates and updates project drafts automatically
- ⛓️ **Blockchain Integration** - Pure Python Solana client with Ed25519 signing
- 🔐 **Secure Signing** - Uses AgentWallet session for transaction signing
- 📋 **Full Logging** - All activity logged with structured format
- ☁️ **Render Ready** - One-click deploy to Render FREE tier

## 📁 Project Structure

```
pow-agent/
├── agent/                   # Core agent logic
│   ├── main.py             # Entry point (daemon)
│   ├── loop.py             # Main async loop
│   ├── heartbeat.py        # Heartbeat checker
│   ├── decision.py         # AI decision engine
│   ├── logger.py           # Structured logging
│   ├── config.py           # Configuration loader
│   └── state.py            # State management
├── colosseum/              # Colosseum API integration
│   ├── api.py              # API wrapper with retry
│   ├── forum.py            # Forum logic
│   ├── project.py          # Project management
│   └── status.py           # Status handling
├── solana/                 # Solana integration
│   ├── client.py           # Pure Python Solana client
│   └── program/            # Anchor program
│       ├── Anchor.toml
│       └── programs/pow_bounty/src/lib.rs
├── data/                   # Persistent state
├── logs/                   # Log files
├── prompts/                # AI prompts
├── tasks/                  # Task definitions
├── Procfile                # Render entry point
├── start.sh                # Startup script
├── render.yaml             # Render Blueprint
└── requirements.txt        # Python dependencies
```

## 🛠️ Prerequisites

### Required Software

1. **Python 3.11+**
```bash
# Windows
winget install Python.Python.3.11

# macOS
brew install python@3.11

# Linux
sudo apt install python3.11 python3.11-venv
```

2. **Rust and Cargo** (for Anchor program deployment)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

3. **Solana CLI**
```bash
sh -c "$(curl -sSfL https://release.solana.com/v1.18.4/install)"
export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
solana --version
```

4. **Anchor CLI**
```bash
cargo install --git https://github.com/coral-xyz/anchor avm --locked --force
avm install latest
avm use latest
anchor --version
```

## ⚡ Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/nayrbryanGaming/proof-of-work-agent.git
cd proof-of-work-agent/pow-agent
```

### 2. Configure Solana Devnet

```bash
# Set to devnet
solana config set --url devnet

# Generate keypair (if needed)
solana-keygen new --outfile ~/.config/solana/id.json

# Get devnet SOL (run multiple times if needed)
solana airdrop 2

# Check balance
solana balance
```

### 3. Deploy Anchor Program

```bash
cd solana/program
anchor build
anchor deploy
# Note the Program ID from output - you'll need this!
```

### 4. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit with your values - REQUIRED:
# - COLOSSEUM_API_KEY
# - OPENAI_API_KEY  
# - AGENTWALLET_SESSION (your Solana keypair)
# - PROGRAM_ID (from anchor deploy)
```

### 5. Run Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the daemon
python agent/main.py
```

## ☁️ Deploy to Render (FREE)

### Option A: One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Option B: Manual Deploy

1. **Push to GitHub**
```bash
git add .
git commit -m "Ready for Render"
git push origin main
```

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Create Background Worker**
   - Click "New" → "Background Worker"
   - Connect your repository
   - Configure:
     - **Name**: `pow-agent-worker`
     - **Runtime**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `bash start.sh`
     - **Plan**: `Free`

4. **Set Environment Variables**
   In Render Dashboard → Environment:
   
   | Variable | Value |
   |----------|-------|
   | `COLOSSEUM_API_KEY` | Your Colosseum API key |
   | `OPENAI_API_KEY` | Your OpenAI API key |
   | `AGENTWALLET_SESSION` | Your Solana keypair JSON |
   | `PROGRAM_ID` | From anchor deploy |
   | `SOLANA_RPC` | `https://api.devnet.solana.com` |
   | `LOOP_INTERVAL_SECONDS` | `1800` |
   | `LOG_LEVEL` | `INFO` |

5. **Deploy**
   - Click "Create Background Worker"
   - Monitor logs in Render dashboard

### Render Blueprint (Auto-Deploy)

The `render.yaml` file enables automatic deployment:
- Push to main branch triggers auto-deploy
- Environment variables configured in blueprint

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `COLOSSEUM_API_KEY` | Colosseum hackathon API key | Yes | - |
| `COLOSSEUM_BASE_URL` | Colosseum API base URL | No | `https://agents.colosseum.com/api` |
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `OPENAI_MODEL` | OpenAI model | No | `gpt-4` |
| `AGENTWALLET_SESSION` | Solana keypair/session | Yes | - |
| `SOLANA_RPC` | Solana RPC endpoint | No | `https://api.devnet.solana.com` |
| `PROGRAM_ID` | Deployed Anchor program ID | Yes | - |
| `HEARTBEAT_URL` | Heartbeat.md URL | No | `https://agents.colosseum.com/heartbeat.md` |
| `LOOP_INTERVAL_SECONDS` | Loop interval | No | `1800` |
| `LOG_LEVEL` | Logging level | No | `INFO` |

### AgentWallet Session

The `AGENTWALLET_SESSION` can be:
- **JSON Array**: `[1,2,3,...,64]` (64 bytes)
- **Base58**: Encoded 64-byte secret key
- **File Path**: `~/.config/solana/id.json`
- **Hex String**: Private key in hex format

Get your keypair JSON:
```bash
cat ~/.config/solana/id.json
# Copy the entire array including brackets [...]
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT LOOP (30 min)                         │
│                     Runs 24/7 as daemon                         │
├─────────────────────────────────────────────────────────────────┤
│  OBSERVE          │  THINK           │  ACT        │  VERIFY   │
│  ─────────        │  ─────           │  ───        │  ──────   │
│  • Heartbeat.md   │  • Decision      │  • Vote     │  • Hash   │
│  • /agents/status │    Engine        │  • Comment  │  • TX     │
│  • Forum Posts    │  • OpenAI        │  • Solve    │  • Log    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SOLANA DEVNET                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  POW_BOUNTY PROGRAM (Anchor)                            │   │
│  │  PDA Seeds: ["bounty", id.to_le_bytes()]                │   │
│  │                                                          │   │
│  │  • create_bounty(id, description, reward)               │   │
│  │  • submit_work(bounty_id, result_hash)                  │   │
│  │  • approve_and_pay(bounty_id)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Cycle Flow

1. **OBSERVE**: Check heartbeat.md and /agents/status
2. **THINK**: Analyze status, decide next action
3. **ACT**: 
   - Engage forum (vote/comment)
   - Solve task with OpenAI
   - Submit proof hash to Solana
4. **VERIFY**: Confirm transaction on-chain
5. **SLEEP**: Wait 30 minutes
6. **REPEAT**

## 📊 Logging

Logs are written to `logs/agent.log` with structured format:
```
[2024-01-15T10:30:45Z][INFO][heartbeat] Heartbeat check completed
[2024-01-15T10:30:46Z][INFO][forum] Voted on post 123
[2024-01-15T10:30:47Z][INFO][solana] TX submitted: 5xK7...abc
```

View logs on Render:
- Dashboard → Your Service → Logs

## 🐛 Troubleshooting

### Common Issues

1. **"PROGRAM_ID is required"**
   - Deploy the Anchor program first: `anchor deploy`
   - Copy the Program ID to your environment variables

2. **"AGENTWALLET_SESSION must be a Solana keypair"**
   - Use JSON array format: `[1,2,3,...,64]`
   - Or base58 encoded secret key
   - Or path to keypair file

3. **"Rate limit exceeded"**
   - The agent has retry logic built in (3 retries)
   - Increase `LOOP_INTERVAL_SECONDS` if needed

4. **Render worker keeps restarting**
   - Check logs for error messages
   - Verify all required environment variables are set
   - Ensure `start.sh` has proper permissions

5. **Solana transaction failing**
   - Get more devnet SOL: `solana airdrop 2`
   - Check RPC endpoint is reachable
   - Verify program is deployed correctly

### Getting Devnet SOL

```bash
# Primary faucet
solana airdrop 2

# Alternative faucets (if rate limited):
# https://faucet.solana.com
# https://solfaucet.com
```

## 📄 License

MIT License - Built for Colosseum Solana Agent Hackathon

---

**Repository**: https://github.com/nayrbryanGaming/proof-of-work-agent

