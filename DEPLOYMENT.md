# 🚀 Deployment Guide - POW Agent

This guide covers deploying the POW Agent to Render.com as a Background Worker (FREE tier).

## Prerequisites

Before deploying, ensure you have:

1. ✅ GitHub repository with the code pushed
2. ✅ Colosseum API key (from agent registration)
3. ✅ OpenAI API key (optional, for AI responses)
4. ✅ Solana wallet with devnet SOL

## 🌐 Deploy to Render

### Method 1: One-Click Deploy (Recommended)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Click the button above
2. Connect your GitHub account
3. Select the `proof-of-work-agent` repository
4. Fill in the environment variables (see below)
5. Click "Create Background Worker"

### Method 2: Manual Deploy

1. Go to [render.com](https://render.com) and sign in
2. Click **New** → **Background Worker**
3. Connect your GitHub repository
4. Configure as follows:

| Setting | Value |
|---------|-------|
| Name | `pow-agent` |
| Region | Oregon (or closest) |
| Branch | `main` |
| Root Directory | `pow-agent` (if in subdirectory) |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `bash start.sh` |
| Plan | Free |

5. Add environment variables (see section below)
6. Click **Create Background Worker**

## 🔐 Environment Variables

Add these in Render Dashboard → Your Service → Environment:

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `COLOSSEUM_API_KEY` | Your Colosseum API key | `75bb47c5e9...` |
| `AGENTWALLET_SESSION` | Solana keypair as JSON array | `[123,45,...]` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for AI responses | (fallback mode) |
| `SOLANA_RPC` | Solana RPC URL | `https://api.devnet.solana.com` |
| `PROGRAM_ID` | Deployed Anchor program ID | (test mode) |
| `LOOP_INTERVAL_SECONDS` | Agent loop interval | `1800` (30 min) |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `ENVIRONMENT` | Runtime environment | `production` |

## 📋 Getting Your Credentials

### 1. Colosseum API Key

If you haven't registered yet:
```bash
cd pow-agent
python scripts/complete_setup.py
```

Or manually via the Colosseum platform.

### 2. AGENTWALLET_SESSION

This is your Solana keypair as a JSON array (64 bytes).

**Option A: Use the setup script**
```bash
python scripts/complete_setup.py
# Check .env for AGENTWALLET_SESSION value
```

**Option B: From Solana CLI**
```bash
solana-keygen new --outfile ~/.config/solana/id.json
cat ~/.config/solana/id.json
# Copy the entire array [1,2,3,...]
```

### 3. Fund Your Wallet

Get devnet SOL for transaction fees:

1. Get your wallet address:
   ```bash
   solana address
   ```

2. Request airdrop:
   ```bash
   solana airdrop 2
   ```
   
   Or use the web faucet: https://faucet.solana.com

## 🔍 Monitoring

### View Logs

In Render Dashboard:
1. Go to your service
2. Click **Logs** tab
3. See real-time agent activity

### Expected Log Output

```
╔═══════════════════════════════════════════════════════════════╗
║                    PROOF-OF-WORK AGENT                        ║
║              Colosseum Solana Agent Hackathon                 ║
╚═══════════════════════════════════════════════════════════════╝

[2026-02-05T10:00:00Z][INFO][main] Starting Proof-of-Work Agent...
[2026-02-05T10:00:01Z][INFO][loop] OBSERVE: heartbeat
[2026-02-05T10:00:02Z][INFO][loop] OBSERVE: agent status
[2026-02-05T10:00:03Z][INFO][loop] ACT: forum engagement
[2026-02-05T10:00:05Z][INFO][loop] THINK: task solving
[2026-02-05T10:00:10Z][INFO][loop] Cycle completed in 10.2s
```

### Health Check

The agent logs a heartbeat every cycle. If you don't see logs for more than 35 minutes, the agent may be down.

## 🔧 Troubleshooting

### Agent Won't Start

1. **Check environment variables** - Ensure `COLOSSEUM_API_KEY` and `AGENTWALLET_SESSION` are set
2. **Check logs** - Look for error messages in Render logs
3. **Validate config** - Run `python scripts/check_config.py` locally

### Rate Limit Errors

The agent has built-in retry logic. If you see frequent 429 errors:
- The agent will automatically back off
- Consider increasing `LOOP_INTERVAL_SECONDS`

### Solana Transaction Failures

1. **Check balance**: Ensure wallet has SOL
2. **Check RPC**: Try a different RPC endpoint
3. **Check program**: Verify PROGRAM_ID is correct (or leave empty for test mode)

### Missing OpenAI Key

The agent works without OpenAI - it uses fallback responses. For better AI reasoning, add `OPENAI_API_KEY`.

## 📊 Metrics & Monitoring

### Built-in Metrics

The agent tracks:
- Cycle success/failure rates
- API response times
- Solana transaction status
- Forum engagement counts

Metrics are saved to `data/metrics.json` (persisted across restarts on Render paid plans).

### External Monitoring

Consider adding:
- **UptimeRobot**: Monitor if service is running
- **Sentry**: Error tracking
- **Datadog**: Full observability

## 🔄 Updates & Redeploy

### Automatic Deploys

Render automatically redeploys when you push to `main`:
1. Make changes locally
2. Commit and push:
   ```bash
   git add -A
   git commit -m "Update agent"
   git push origin main
   ```
3. Render will automatically rebuild and restart

### Manual Redeploy

In Render Dashboard:
1. Go to your service
2. Click **Manual Deploy** → **Deploy latest commit**

## 💰 Cost

### Free Tier Limits

Render Free tier includes:
- 750 hours/month of Background Worker
- Spins down after 15 min of inactivity (doesn't apply to workers)
- Shared CPU and memory

For 24/7 operation, Free tier is sufficient!

### Upgrade Options

If you need more resources:
- **Starter ($7/month)**: Dedicated CPU, never sleeps
- **Standard ($25/month)**: More CPU/memory

## 🔐 Security Best Practices

1. **Never commit secrets** - Use environment variables
2. **Use .gitignore** - Excludes `.env`, logs, etc.
3. **Rotate keys** - Regenerate API keys periodically
4. **Monitor access** - Check logs for unauthorized activity

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Colosseum Hackathon](https://colosseum.com)
- [Solana Devnet Faucet](https://faucet.solana.com)
- [Project GitHub](https://github.com/nayrbryanGaming/proof-of-work-agent)

---

## Need Help?

1. Check the [README.md](README.md) for local development
2. Open an issue on GitHub
3. Join the Colosseum Discord

Happy deploying! 🚀
