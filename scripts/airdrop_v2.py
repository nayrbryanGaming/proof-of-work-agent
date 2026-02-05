import requests
import time
import sys

WALLET = "5JTDJdfDHqu3TEHuBTATJF49G8i8YKy42riJG9KWFfSk"

# More RPC endpoints to try
RPCS = {
    "devnet": [
        "https://api.devnet.solana.com",
        "https://devnet.helius-rpc.com/?api-key=15319bf4-5b40-4958-ac8d-6313aa55eb92",
        "https://rpc-devnet.helius.xyz/?api-key=15319bf4-5b40-4958-ac8d-6313aa55eb92",
    ],
    "testnet": [
        "https://api.testnet.solana.com",
        "https://testnet.solana.com",
        "https://solana-testnet-rpc.publicnode.com",
    ]
}

def main():
    print("SOLANA AIRDROP v2")
    print("="*50)
    print(f"Wallet: {WALLET}")
    
    for network, rpcs in RPCS.items():
        print(f"\n{'='*50}")
        print(f"{network.upper()}")
        print("="*50)
        
        for i, rpc in enumerate(rpcs):
            print(f"\n[{i+1}/{len(rpcs)}] Trying: {rpc[:50]}...")
            
            # Balance
            try:
                r = requests.post(rpc, json={"jsonrpc":"2.0","id":1,"method":"getBalance","params":[WALLET]}, timeout=15, headers={"Content-Type": "application/json"})
                data = r.json()
                if "result" in data:
                    bal = data["result"]["value"] / 1e9
                    print(f"  Balance: {bal:.4f} SOL")
                    
                    if bal >= 1.0:
                        print("  OK - has SOL!")
                        break
                else:
                    print(f"  Balance failed: {data.get('error', 'Unknown')}")
                    continue
            except Exception as e:
                print(f"  Balance error: {e}")
                continue
            
            # Airdrop - try smaller amounts
            for amount in [1000000000, 500000000]:  # 1 SOL, 0.5 SOL
                print(f"  Requesting {amount/1e9} SOL...")
                try:
                    r = requests.post(rpc, 
                        json={"jsonrpc":"2.0","id":1,"method":"requestAirdrop","params":[WALLET, amount]}, 
                        timeout=60,
                        headers={"Content-Type": "application/json"}
                    )
                    data = r.json()
                    
                    if "result" in data:
                        sig = data["result"]
                        print(f"  TX: {sig[:50]}...")
                        print("  Waiting for confirmation...")
                        
                        # Wait and check
                        time.sleep(10)
                        
                        r = requests.post(rpc, json={"jsonrpc":"2.0","id":1,"method":"getBalance","params":[WALLET]}, timeout=10)
                        new_data = r.json()
                        if "result" in new_data:
                            new_bal = new_data["result"]["value"] / 1e9
                            print(f"  New Balance: {new_bal:.4f} SOL")
                            if new_bal > bal:
                                print("  SUCCESS!")
                                break
                    else:
                        err = data.get("error", {})
                        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        print(f"  Failed: {msg[:80]}")
                        
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                continue
            break  # Airdrop succeeded, move to next network
    
    # Final summary
    print(f"\n{'='*50}")
    print("FINAL BALANCES")
    print("="*50)
    
    for network, rpcs in RPCS.items():
        try:
            r = requests.post(rpcs[0], json={"jsonrpc":"2.0","id":1,"method":"getBalance","params":[WALLET]}, timeout=10)
            data = r.json()
            if "result" in data:
                bal = data["result"]["value"] / 1e9
                status = "OK" if bal > 0 else "EMPTY"
                print(f"  {network}: {bal:.4f} SOL [{status}]")
            else:
                print(f"  {network}: ERROR")
        except:
            print(f"  {network}: UNREACHABLE")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
