import requests
import time
import sys

WALLET = "5JTDJdfDHqu3TEHuBTATJF49G8i8YKy42riJG9KWFfSk"

def main():
    sys.stdout.write("SOLANA AIRDROP\n")
    sys.stdout.write("="*50 + "\n")
    sys.stdout.flush()
    
    networks = [
        ("devnet", "https://api.devnet.solana.com"),
        ("testnet", "https://api.testnet.solana.com"),
        ("testnet-ankr", "https://rpc.ankr.com/solana_testnet"),
    ]
    
    for name, rpc in networks:
        sys.stdout.write(f"\n{name.upper()}:\n")
        sys.stdout.flush()
        
        # Balance
        try:
            r = requests.post(rpc, json={"jsonrpc":"2.0","id":1,"method":"getBalance","params":[WALLET]}, timeout=15)
            data = r.json()
            if "result" in data:
                bal = data["result"]["value"] / 1e9
                sys.stdout.write(f"  Balance: {bal:.4f} SOL\n")
                sys.stdout.flush()
                
                if bal >= 2:
                    sys.stdout.write("  Sufficient balance\n")
                    sys.stdout.flush()
                    continue
            else:
                sys.stdout.write(f"  Balance error: {data}\n")
                sys.stdout.flush()
                continue
        except Exception as e:
            sys.stdout.write(f"  Error: {e}\n")
            sys.stdout.flush()
            continue
        
        # Airdrop
        sys.stdout.write("  Requesting airdrop...\n")
        sys.stdout.flush()
        
        try:
            r = requests.post(rpc, json={"jsonrpc":"2.0","id":1,"method":"requestAirdrop","params":[WALLET, 2000000000]}, timeout=30)
            data = r.json()
            
            if "result" in data:
                sig = data["result"]
                sys.stdout.write(f"  TX: {sig[:40]}...\n")
                sys.stdout.flush()
                
                sys.stdout.write("  Waiting 5s...\n")
                sys.stdout.flush()
                time.sleep(5)
                
                # New balance
                r = requests.post(rpc, json={"jsonrpc":"2.0","id":1,"method":"getBalance","params":[WALLET]}, timeout=10)
                data = r.json()
                if "result" in data:
                    new_bal = data["result"]["value"] / 1e9
                    sys.stdout.write(f"  New Balance: {new_bal:.4f} SOL\n")
                    sys.stdout.flush()
            elif "error" in data:
                err = data["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                sys.stdout.write(f"  Airdrop Error: {msg}\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"  Airdrop Exception: {e}\n")
            sys.stdout.flush()
    
    sys.stdout.write("\nDone!\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
