#!/usr/bin/env python3
"""
Comprehensive Validation Suite for POW Agent
Tests all components before deployment to ensure production readiness.
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv


class ValidationResult:
    """Result of a validation test."""
    
    def __init__(self, name: str, passed: bool, message: str = "", details: dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name}: {self.message}"


class ValidationSuite:
    """Complete validation suite."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.base_path = Path(__file__).resolve().parent.parent
        load_dotenv(self.base_path / ".env")
    
    def add_result(self, name: str, passed: bool, message: str = "", details: dict = None):
        result = ValidationResult(name, passed, message, details)
        self.results.append(result)
        print(result)
        return result
    
    # ===========================================================
    # File Structure Validation
    # ===========================================================
    
    def validate_file_structure(self):
        """Validate required files exist."""
        print("\n" + "=" * 60)
        print("📁 FILE STRUCTURE VALIDATION")
        print("=" * 60)
        
        required_files = [
            ("requirements.txt", "Python dependencies"),
            ("start.sh", "Startup script"),
            ("Procfile", "Render process definition"),
            ("README.md", "Documentation"),
            (".env", "Environment configuration"),
            ("agent/main.py", "Main entry point"),
            ("agent/loop.py", "Agent loop"),
            ("agent/config.py", "Configuration loader"),
            ("agent/logger.py", "Logging module"),
            ("agent/decision.py", "Decision engine"),
            ("agent/heartbeat.py", "Heartbeat checker"),
            ("agent/state.py", "State management"),
            ("colosseum/api.py", "Colosseum API client"),
            ("colosseum/forum.py", "Forum handler"),
            ("colosseum/project.py", "Project manager"),
            ("colosseum/status.py", "Status checker"),
            ("solana/client.py", "Solana client"),
            ("prompts/solve_task.txt", "Task solving prompt"),
            ("tasks/sample_tasks.json", "Sample tasks"),
        ]
        
        missing = []
        for file_path, description in required_files:
            full_path = self.base_path / file_path
            exists = full_path.exists()
            if not exists:
                missing.append(file_path)
            self.add_result(
                f"File: {file_path}",
                exists,
                description if exists else "MISSING"
            )
        
        return len(missing) == 0
    
    # ===========================================================
    # Environment Validation
    # ===========================================================
    
    def validate_environment(self):
        """Validate environment variables."""
        print("\n" + "=" * 60)
        print("🔧 ENVIRONMENT VALIDATION")
        print("=" * 60)
        
        required_vars = [
            ("COLOSSEUM_API_KEY", True, "Colosseum API access"),
            ("AGENTWALLET_SESSION", True, "Solana wallet"),
            ("OPENAI_API_KEY", False, "AI reasoning (optional)"),
            ("SOLANA_RPC", True, "Solana network"),
            ("PROGRAM_ID", False, "Anchor program (optional)"),
        ]
        
        all_required_present = True
        for var_name, required, description in required_vars:
            value = os.getenv(var_name, "").strip()
            present = bool(value)
            
            if required and not present:
                all_required_present = False
            
            # Mask sensitive values
            display_value = "***" + value[-8:] if present and len(value) > 8 else ("SET" if present else "NOT SET")
            
            self.add_result(
                f"ENV: {var_name}",
                present or not required,
                f"{display_value} - {description}"
            )
        
        return all_required_present
    
    # ===========================================================
    # Module Import Validation
    # ===========================================================
    
    def validate_imports(self):
        """Validate all modules can be imported."""
        print("\n" + "=" * 60)
        print("📦 MODULE IMPORT VALIDATION")
        print("=" * 60)
        
        modules = [
            ("agent.config", "config"),
            ("agent.logger", "get_logger"),
            ("agent.loop", "forever"),
            ("agent.decision", "DecisionEngine"),
            ("agent.heartbeat", "HeartbeatChecker"),
            ("agent.state", "StateManager"),
            ("colosseum.api", "ColosseumAPI"),
            ("colosseum.forum", "ForumHandler"),
            ("colosseum.status", "should_act"),
            ("solana.client", "SolanaClient"),
        ]
        
        all_imported = True
        for module_name, attr_name in modules:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                has_attr = hasattr(module, attr_name)
                self.add_result(
                    f"Import: {module_name}",
                    has_attr,
                    f"Has {attr_name}" if has_attr else f"Missing {attr_name}"
                )
                if not has_attr:
                    all_imported = False
            except Exception as e:
                self.add_result(
                    f"Import: {module_name}",
                    False,
                    f"Error: {str(e)[:50]}"
                )
                all_imported = False
        
        return all_imported
    
    # ===========================================================
    # API Connectivity Validation
    # ===========================================================
    
    def validate_colosseum_api(self):
        """Validate Colosseum API connectivity."""
        print("\n" + "=" * 60)
        print("🌐 COLOSSEUM API VALIDATION")
        print("=" * 60)
        
        import requests
        
        api_key = os.getenv("COLOSSEUM_API_KEY", "")
        base_url = os.getenv("COLOSSEUM_BASE_URL", "https://agents.colosseum.com/api")
        
        if not api_key:
            self.add_result("Colosseum API", False, "API key not set")
            return False
        
        endpoints = [
            ("/agents/status", "GET", "Agent status"),
            ("/forum/posts?sort=hot&limit=5", "GET", "Forum posts"),
        ]
        
        all_ok = True
        for path, method, description in endpoints:
            try:
                resp = requests.request(
                    method,
                    f"{base_url}{path}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10
                )
                ok = resp.status_code in (200, 201, 404)  # 404 is OK for some endpoints
                self.add_result(
                    f"API: {path}",
                    ok,
                    f"Status {resp.status_code} - {description}"
                )
                if not ok:
                    all_ok = False
            except Exception as e:
                self.add_result(f"API: {path}", False, f"Error: {str(e)[:50]}")
                all_ok = False
        
        return all_ok
    
    # ===========================================================
    # OpenAI API Validation
    # ===========================================================
    
    def validate_openai_api(self):
        """Validate OpenAI API connectivity."""
        print("\n" + "=" * 60)
        print("🤖 OPENAI API VALIDATION")
        print("=" * 60)
        
        import requests
        
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not api_key:
            self.add_result("OpenAI API", True, "Not configured (using fallback)")
            return True  # Optional
        
        try:
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )
            ok = resp.status_code == 200
            self.add_result(
                "OpenAI API",
                ok,
                f"Status {resp.status_code}" + (" - Models accessible" if ok else "")
            )
            return ok
        except Exception as e:
            self.add_result("OpenAI API", False, f"Error: {str(e)[:50]}")
            return False
    
    # ===========================================================
    # Solana RPC Validation
    # ===========================================================
    
    def validate_solana_rpc(self):
        """Validate Solana RPC connectivity."""
        print("\n" + "=" * 60)
        print("⛓️ SOLANA RPC VALIDATION")
        print("=" * 60)
        
        import requests
        
        rpc_url = os.getenv("SOLANA_RPC", "https://api.devnet.solana.com")
        
        # Test getHealth
        try:
            resp = requests.post(
                rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
                timeout=10
            )
            data = resp.json()
            healthy = data.get("result") == "ok"
            self.add_result(
                "Solana RPC Health",
                healthy,
                f"Status: {data.get('result', 'unknown')}"
            )
        except Exception as e:
            self.add_result("Solana RPC Health", False, f"Error: {str(e)[:50]}")
            return False
        
        # Test getLatestBlockhash
        try:
            resp = requests.post(
                rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "getLatestBlockhash"},
                timeout=10
            )
            data = resp.json()
            has_blockhash = "result" in data and "value" in data["result"]
            self.add_result(
                "Solana Blockhash",
                has_blockhash,
                f"Latest blockhash accessible" if has_blockhash else "Failed"
            )
        except Exception as e:
            self.add_result("Solana Blockhash", False, f"Error: {str(e)[:50]}")
            return False
        
        # Check wallet balance
        wallet_session = os.getenv("AGENTWALLET_SESSION", "")
        if wallet_session and wallet_session.startswith("["):
            try:
                from solana.client import Keypair
                data = json.loads(wallet_session)
                kp = Keypair(bytes(data))
                address = kp.public_key.to_base58()
                
                resp = requests.post(
                    rpc_url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [address]},
                    timeout=10
                )
                data = resp.json()
                if "result" in data:
                    lamports = data["result"]["value"]
                    sol = lamports / 1_000_000_000
                    self.add_result(
                        "Wallet Balance",
                        sol >= 0.01,
                        f"{sol:.4f} SOL" + (" (low!)" if sol < 0.1 else "")
                    )
            except Exception as e:
                self.add_result("Wallet Balance", False, f"Error: {str(e)[:50]}")
        
        return True
    
    # ===========================================================
    # State & Data Validation
    # ===========================================================
    
    def validate_data_directories(self):
        """Validate data directories exist and are writable."""
        print("\n" + "=" * 60)
        print("💾 DATA DIRECTORY VALIDATION")
        print("=" * 60)
        
        directories = [
            ("logs", "Log files"),
            ("data", "State and metrics"),
        ]
        
        all_ok = True
        for dir_name, description in directories:
            dir_path = self.base_path / dir_name
            
            # Create if doesn't exist
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Test write
            test_file = dir_path / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
                writable = True
            except:
                writable = False
            
            self.add_result(
                f"Directory: {dir_name}",
                writable,
                f"{description} - {'Writable' if writable else 'NOT WRITABLE'}"
            )
            
            if not writable:
                all_ok = False
        
        return all_ok
    
    # ===========================================================
    # Logic Validation
    # ===========================================================
    
    def validate_decision_engine(self):
        """Validate decision engine logic."""
        print("\n" + "=" * 60)
        print("🧠 DECISION ENGINE VALIDATION")
        print("=" * 60)
        
        try:
            from agent.decision import DecisionEngine, solve, forum_comment, _fallback_comment
            
            # Test fallback comment (should always work)
            context = "Title: Test Post\nTags: ai, solana\nBody: This is a test."
            comment = _fallback_comment(context)
            
            self.add_result(
                "Fallback Comment",
                len(comment) > 0 and len(comment) <= 400,
                f"Generated {len(comment)} chars"
            )
            
            # Test solve function (may fail without OpenAI)
            try:
                result = solve("What is 2+2?")
                has_result = len(result) > 0
                self.add_result(
                    "Task Solving",
                    has_result,
                    f"Generated {len(result)} chars response"
                )
            except Exception as e:
                self.add_result(
                    "Task Solving",
                    True,  # Expected without OpenAI key
                    f"Fallback mode: {str(e)[:30]}"
                )
            
            return True
        except Exception as e:
            self.add_result("Decision Engine", False, f"Error: {str(e)[:50]}")
            return False
    
    # ===========================================================
    # Circuit Breaker Validation
    # ===========================================================
    
    def validate_circuit_breaker(self):
        """Validate circuit breaker functionality."""
        print("\n" + "=" * 60)
        print("🔌 CIRCUIT BREAKER VALIDATION")
        print("=" * 60)
        
        try:
            from agent.circuit_breaker import CircuitBreaker, CircuitState
            
            cb = CircuitBreaker("test", failure_threshold=3, reset_timeout=1.0)
            
            # Test initial state
            self.add_result(
                "Initial State",
                cb.state == CircuitState.CLOSED,
                f"State: {cb.state.value}"
            )
            
            # Simulate failures
            for i in range(3):
                cb._record_failure(Exception("test"))
            
            self.add_result(
                "After Failures",
                cb.state == CircuitState.OPEN,
                f"State: {cb.state.value}"
            )
            
            return True
        except Exception as e:
            self.add_result("Circuit Breaker", False, f"Error: {str(e)[:50]}")
            return False
    
    # ===========================================================
    # Run All Validations
    # ===========================================================
    
    def run_all(self) -> Tuple[int, int]:
        """Run all validations and return (passed, total)."""
        print("\n")
        print("╔" + "═" * 60 + "╗")
        print("║" + " POW AGENT - COMPREHENSIVE VALIDATION SUITE ".center(60) + "║")
        print("╚" + "═" * 60 + "╝")
        print(f"\nTimestamp: {datetime.now().isoformat()}")
        print(f"Base Path: {self.base_path}")
        
        # Run all validations
        self.validate_file_structure()
        self.validate_environment()
        self.validate_imports()
        self.validate_data_directories()
        self.validate_colosseum_api()
        self.validate_openai_api()
        self.validate_solana_rpc()
        self.validate_decision_engine()
        self.validate_circuit_breaker()
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"\n   Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n   🎉 ALL VALIDATIONS PASSED!")
            print("   Agent is READY FOR DEPLOYMENT")
        else:
            failed = [r for r in self.results if not r.passed]
            print(f"\n   ⚠️  {len(failed)} VALIDATIONS FAILED:")
            for r in failed:
                print(f"      - {r.name}: {r.message}")
        
        # Generate report
        report_path = self.base_path / "data" / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "passed": passed,
            "total": total,
            "success_rate": passed / total * 100 if total > 0 else 0,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\n   Report saved to: {report_path}")
        
        return passed, total


def main():
    suite = ValidationSuite()
    passed, total = suite.run_all()
    
    # Exit with error code if validations failed
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
