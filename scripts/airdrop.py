#!/usr/bin/env python3
"""
Solana Airdrop Script - Pure Python (No CLI Required)
Requests SOL airdrop on devnet and testnet.
"""

import json
import sys
import time
import requests
from typing import Optional

# Your wallet address
WALLET_ADDRESS = "5JTDJdfDHqu3TEHuBTATJF49G8i8YKy42riJG9KWFfSk"

# RPC Endpoints
RPCS = {
    "devnet": [
        "https://api.devnet.solana.com",
        "https://devnet.helius-rpc.com/?api-key=15319bf4-5b40-4958-ac8d-6313aa55eb92",
    ],
    "testnet": [
        "https://api.testnet.solana.com",
        "https://rpc.ankr.com/solana_testnet",
        "https://testnet.solana.com",
    ]
}


def get_balance(rpc_url: str, address: str) -> Optional[float]:
    """Get SOL balance for an address."""
    try:
        resp = requests.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [address]
            },
            timeout=10
        )
        data = resp.json()
        if "result" in data:
            lamports = data["result"]["value"]
            return lamports / 1_000_000_000
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def request_airdrop(rpc_url: str, address: str, amount_sol: float = 2.0) -> Optional[str]:
    """Request airdrop from Solana network."""
    lamports = int(amount_sol * 1_000_000_000)
    
    try:
        resp = requests.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "requestAirdrop",
                "params": [address, lamports]
            },
            timeout=30
        )
        data = resp.json()
        
        if "result" in data:
            return data["result"]  # Transaction signature
        elif "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                print(f"  Error: {error.get('message', error)}")
            else:
                print(f"  Error: {error}")
            return None
        return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None


def confirm_transaction(rpc_url: str, signature: str, max_retries: int = 30) -> bool:
    """Wait for transaction confirmation."""
    for i in range(max_retries):
        try:
            resp = requests.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignatureStatuses",
                    "params": [[signature]]
                },
                timeout=10
            )
            data = resp.json()
            
            if "result" in data and data["result"]["value"]:
                status = data["result"]["value"][0]
                if status and status.get("confirmationStatus") in ["confirmed", "finalized"]:
                    return True
        except:
            pass
        
        time.sleep(1)
    
    return False


def main():
    print("=" * 60)
    print("  SOLANA AIRDROP - Pure Python")
    print("=" * 60)
    print(f"\n📍 Wallet: {WALLET_ADDRESS}\n")
    
    networks = ["devnet", "testnet"]
    
    for network in networks:
        print(f"\n{'='*60}")
        print(f"🌐 {network.upper()}")
        print("=" * 60)
        
        rpcs = RPCS.get(network, [])
        success = False
        
        for rpc in rpcs:
            print(f"\n📡 Trying RPC: {rpc[:50]}...")
            
            # Check current balance
            balance = get_balance(rpc, WALLET_ADDRESS)
            if balance is not None:
                print(f"💰 Current balance: {balance:.4f} SOL")
            else:
                print("⚠️  Could not fetch balance")
                continue
            
            # Skip if already has enough
            if balance >= 2.0:
                print(f"✅ Already has {balance:.4f} SOL, skipping airdrop")
                success = True
                break
            
            # Request airdrop
            print(f"🚰 Requesting 2 SOL airdrop...")
            sig = request_airdrop(rpc, WALLET_ADDRESS, 2.0)
            
            if sig:
                print(f"📝 Transaction: {sig[:20]}...")
                print("⏳ Waiting for confirmation...")
                
                if confirm_transaction(rpc, sig):
                    new_balance = get_balance(rpc, WALLET_ADDRESS)
                    print(f"✅ Airdrop confirmed!")
                    print(f"💰 New balance: {new_balance:.4f} SOL")
                    success = True
                    break
                else:
                    print("⚠️  Confirmation timeout, trying next RPC...")
            else:
                print("❌ Airdrop request failed, trying next RPC...")
        
        if success:
            print(f"\n✅ {network.upper()} airdrop successful!")
        else:
            print(f"\n❌ {network.upper()} airdrop failed on all RPCs")
            print(f"   Manual faucet: https://faucet.solana.com (select {network})")
    
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    for network in networks:
        rpc = RPCS[network][0]
        balance = get_balance(rpc, WALLET_ADDRESS)
        status = "✅" if balance and balance >= 0.1 else "❌"
        print(f"  {status} {network.upper()}: {balance:.4f if balance else 0:.4f} SOL")
    
    print("\n")


if __name__ == "__main__":
    main()
