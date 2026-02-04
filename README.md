# Proof-of-Work Agent

![POW Agent](https://img.shields.io/badge/POW-Agent-blue)
![Solana](https://img.shields.io/badge/Solana-Devnet-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An autonomous AI agent for the Colosseum Solana Agent Hackathon that demonstrates the **observe → think → act → verify** loop. Complete with production-ready API backend and real-time dashboard.

## 🚀 Features

- 🔄 **Autonomous Loop** - Polls heartbeat.md every 30 minutes
- 📊 **Decision Engine** - Uses `/agents/status` as decision signal
- 💬 **Forum Engagement** - Reads, votes, and comments meaningfully on forum posts
- 📝 **Project Management** - Creates and updates project drafts automatically
- ⛓️ **Blockchain Integration** - Deploys and uses Anchor bounty program on Solana devnet
- 🔐 **Secure Signing** - Uses AgentWallet session for transaction signing
- 📋 **Full Logging** - All activity logged with structured format
- 🎛️ **Dashboard** - Real-time monitoring dashboard with WebSocket updates
- 🔧 **REST API** - Complete API for control and monitoring
- 🐳 **Docker Ready** - Containerized for easy deployment
- ☁️ **Vercel Ready** - Dashboard deployable to Vercel

## 📁 Project Structure

```
pow-agent/
├── agent/                   # Core agent logic
│   ├── main.py             # Entry point
│   ├── loop.py             # Main async loop
│   ├── heartbeat.py        # Heartbeat checker
│   ├── decision.py         # AI decision engine
│   ├── logger.py           # Structured logging
│   ├── config.py           # Configuration loader
│   ├── state.py            # State management
│   ├── validators.py       # Input validation
│   └── circuit_breaker.py  # Fault tolerance
├── api/                    # FastAPI backend
│   ├── server.py           # API server
│   ├── routes.py           # REST endpoints
│   └── websocket.py        # WebSocket manager
├── colosseum/              # Colosseum API integration
│   ├── api.py              # API wrapper
│   ├── forum.py            # Forum logic
│   ├── project.py          # Project management
│   └── status.py           # Status handling
├── solana/                 # Solana integration
│   ├── client.py           # Transaction client
│   └── program/            # Anchor program
│       ├── Anchor.toml
│       └── programs/pow_bounty/src/lib.rs
├── dashboard/              # Next.js frontend
│   ├── src/app/           # App router pages
│   ├── src/components/    # React components
│   └── src/hooks/         # Custom hooks
├── data/                   # Persistent state
├── logs/                   # Log files
├── prompts/                # AI prompts
├── tasks/                  # Task definitions
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Backend container
├── vercel.json             # Vercel config
└── requirements.txt        # Python dependencies
```

## 🛠️ Prerequisites

### Required Software

1. **Python 3.10+**
```bash
# Windows
winget install Python.Python.3.11

# macOS
brew install python@3.11

# Linux
sudo apt install python3.11 python3.11-venv
```

2. **Node.js 18+** (for dashboard)
```bash
# Windows
winget install OpenJS.NodeJS.LTS

# macOS
brew install node@20

# Linux
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs
```

3. **Rust and Cargo** (for Anchor)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

4. **Solana CLI**
```bash
sh -c "$(curl -sSfL https://release.solana.com/v1.18.4/install)"
export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
solana --version
```

5. **Anchor CLI**
```bash
cargo install --git https://github.com/coral-xyz/anchor avm --locked --force
avm install latest
avm use latest
anchor --version
```

## ⚡ Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/pow-agent.git
cd pow-agent
```

### 2. Configure Solana

```bash
# Set to devnet
solana config set --url devnet

# Generate keypair (if needed)
solana-keygen new --outfile ~/.config/solana/id.json

# Get devnet SOL
solana airdrop 2
```

### 3. Deploy Anchor Program

```bash
cd solana/program
anchor build
anchor deploy
# Note the Program ID from output
```

### 4. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit with your values
# Required:
#   COLOSSEUM_API_KEY
#   OPENAI_API_KEY
#   AGENTWALLET_SESSION
#   PROGRAM_ID (from step 3)
```

### 5. Install Dependencies

```bash
# Python backend
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Dashboard
cd dashboard
npm install
cd ..
```

### 6. Run the Agent

**Option A: Agent only**
```bash
python agent/main.py
```

**Option B: With API + Dashboard**
```bash
# Terminal 1: API Server
python run_api.py

# Terminal 2: Dashboard
cd dashboard
npm run dev

# Open http://localhost:3000
```

**Option C: Docker**
```bash
docker-compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:3000
```

## ☁️ Deploy to Vercel

### Deploy Dashboard

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/pow-agent.git
git push -u origin main
```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Import your repository
   - Set root directory to `dashboard`
   - Add environment variables:
     - `NEXT_PUBLIC_API_URL` = Your API URL
     - `NEXT_PUBLIC_WS_URL` = Your WebSocket URL

3. **Deploy API Backend**
   - Deploy to Railway, Render, or any Python hosting
   - Or use the provided Dockerfile

### Environment Variables for Vercel

```env
NEXT_PUBLIC_API_URL=https://your-api-domain.com
NEXT_PUBLIC_WS_URL=wss://your-api-domain.com
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `COLOSSEUM_API_KEY` | Colosseum hackathon API key | Yes | - |
| `COLOSSEUM_BASE_URL` | Colosseum API base URL | No | `https://colosseum.com/api` |
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `OPENAI_MODEL` | OpenAI model | No | `gpt-4` |
| `AGENTWALLET_SESSION` | Solana keypair/session | Yes | - |
| `SOLANA_RPC` | Solana RPC endpoint | No | `https://api.devnet.solana.com` |
| `PROGRAM_ID` | Deployed Anchor program ID | Yes | - |
| `HEARTBEAT_URL` | Heartbeat.md URL | No | `https://colosseum.com/heartbeat.md` |
| `LOOP_INTERVAL_SECONDS` | Loop interval | No | `1800` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `API_HOST` | API server host | No | `0.0.0.0` |
| `API_PORT` | API server port | No | `8000` |

### AgentWallet Session

The `AGENTWALLET_SESSION` can be:
- Base58 encoded 64-byte secret key
- JSON array of bytes
- Path to a keypair file (e.g., `~/.config/solana/id.json`)

## 📡 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/status` | Agent status |
| `GET` | `/api/metrics` | Agent metrics |
| `GET` | `/api/cycles` | Recent cycles |
| `GET` | `/api/logs` | Agent logs |
| `GET` | `/api/tasks` | Task queue |
| `GET` | `/api/config` | Configuration |
| `POST` | `/api/start` | Start agent |
| `POST` | `/api/stop` | Stop agent |
| `POST` | `/api/trigger-cycle` | Manual cycle |
| `POST` | `/api/tasks` | Create task |
| `DELETE` | `/api/tasks/{id}` | Delete task |

### WebSocket

Connect to `/api/ws` for real-time updates:
- `cycle_complete` - Cycle finished
- `state_update` - State changed
- `log` - New log entry
- `agent_started` - Agent started
- `agent_stopped` - Agent stopped

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DASHBOARD (Next.js)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Metrics  │  │   Logs   │  │  Tasks   │  │  Config  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
└───────┼─────────────┼─────────────┼─────────────┼──────────────┘
        │             │             │             │
        └─────────────┴──────┬──────┴─────────────┘
                             │
                    ┌────────┴────────┐
                    │  FastAPI + WS   │
                    │   REST API      │
                    └────────┬────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                     AGENT LOOP (30 min)                         │
├─────────────────────────────────────────────────────────────────┤
│  OBSERVE          │  THINK           │  ACT        │  VERIFY   │
│  ─────────        │  ─────           │  ───        │  ──────   │
│  • Heartbeat      │  • Decision      │  • Vote     │  • Hash   │
│  • Status API     │    Engine        │  • Comment  │  • TX     │
│  • Forum Posts    │  • AI Analysis   │  • Solve    │  • Log    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SOLANA DEVNET                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  POW_BOUNTY PROGRAM                                      │   │
│  │  • create_bounty(id, description, reward)               │   │
│  │  • submit_work(bounty_id, result_hash)                  │   │
│  │  • approve_and_pay(bounty_id)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Logging

Logs are written to `logs/agent.log` with structured format:
```
[2024-01-15T10:30:45Z][INFO][heartbeat] Heartbeat check completed
[2024-01-15T10:30:46Z][INFO][forum] Voted on post 123
[2024-01-15T10:30:47Z][INFO][solana] TX submitted: 5xK7...abc
```

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=agent --cov=colosseum --cov=solana

# Type checking
mypy agent colosseum solana
```

## 🐛 Troubleshooting

### Common Issues

1. **"PROGRAM_ID is required"**
   - Deploy the Anchor program first: `anchor deploy`
   - Copy the Program ID to `.env`

2. **"AGENTWALLET_SESSION must be a Solana keypair"**
   - Use base58 encoded secret key
   - Or path to keypair file: `~/.config/solana/id.json`

3. **"Rate limit exceeded"**
   - The agent has retry logic built in
   - Reduce `LOOP_INTERVAL_SECONDS` if needed

4. **WebSocket connection failed**
   - Check API server is running
   - Verify `NEXT_PUBLIC_WS_URL` is correct

## 📄 License

MIT License - Built for Colosseum Solana Agent Hackathon

