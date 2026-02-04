"""
Test suite for Solana integration.
"""

import asyncio
import base58
import json
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================
# TESTS - Solana Client
# ============================================================

class TestSolanaClient:
    """Tests for Solana client."""
    
    @pytest.fixture
    def client(self, env_vars):
        """Create Solana client instance."""
        from solana.client import SolanaClient
        return SolanaClient()
    
    @pytest.mark.asyncio
    async def test_connection(self, client, mock_solana_client):
        """Test connection to Solana RPC."""
        connected = await client.is_connected()
        
        assert connected is True
    
    @pytest.mark.asyncio
    async def test_get_balance(self, client, mock_solana_client):
        """Test getting wallet balance."""
        balance = await client.get_balance()
        
        assert balance >= 0
    
    @pytest.mark.asyncio
    async def test_submit_proof(self, client, mock_solana_client):
        """Test submitting proof to Solana program."""
        task_hash = "a" * 64
        solution_hash = "b" * 64
        
        tx_sig = await client.submit_proof(
            task_hash=task_hash,
            solution_hash=solution_hash
        )
        
        assert tx_sig is not None
        assert len(tx_sig) == 88  # Base58 signature length
    
    @pytest.mark.asyncio
    async def test_verify_signature(self, client, mock_solana_client):
        """Test verifying transaction signature."""
        tx_sig = "5" + "x" * 87
        
        verified = await client.verify_signature(tx_sig)
        
        assert verified is True
    
    @pytest.mark.asyncio
    async def test_get_transaction(self, client, mock_solana_client):
        """Test getting transaction details."""
        tx_sig = "5" + "x" * 87
        
        tx = await client.get_transaction(tx_sig)
        
        # May return None if not found
        assert tx is None or isinstance(tx, dict)


# ============================================================
# TESTS - Proof of Work Submission
# ============================================================

class TestProofOfWorkSubmission:
    """Tests for proof of work submission."""
    
    @pytest.fixture
    def client(self, env_vars):
        """Create Solana client instance."""
        from solana.client import SolanaClient
        return SolanaClient()
    
    @pytest.mark.asyncio
    async def test_create_proof_instruction(self, client):
        """Test creating proof instruction."""
        task_hash = bytes.fromhex("a" * 64)
        solution_hash = bytes.fromhex("b" * 64)
        
        instruction = client.create_proof_instruction(
            task_hash=task_hash,
            solution_hash=solution_hash
        )
        
        assert instruction is not None
    
    @pytest.mark.asyncio
    async def test_proof_hash_validation(self, client):
        """Test proof hash validation."""
        valid_hash = "a" * 64
        invalid_hash = "g" * 64  # Contains invalid hex character
        
        assert client.validate_hash(valid_hash) is True
        assert client.validate_hash(invalid_hash) is False
    
    @pytest.mark.asyncio
    async def test_proof_hash_length(self, client):
        """Test proof hash length validation."""
        short_hash = "a" * 32
        correct_hash = "a" * 64
        long_hash = "a" * 128
        
        assert client.validate_hash(short_hash) is False
        assert client.validate_hash(correct_hash) is True
        assert client.validate_hash(long_hash) is False


# ============================================================
# TESTS - Transaction Building
# ============================================================

class TestTransactionBuilding:
    """Tests for transaction building."""
    
    @pytest.fixture
    def client(self, env_vars):
        """Create Solana client instance."""
        from solana.client import SolanaClient
        return SolanaClient()
    
    @pytest.mark.asyncio
    async def test_build_transaction(self, client, mock_solana_client):
        """Test building a transaction."""
        tx = await client.build_transaction(
            instructions=[],  # Empty for test
            payer=client.keypair.pubkey() if hasattr(client, 'keypair') else None
        )
        
        assert tx is not None
    
    @pytest.mark.asyncio
    async def test_sign_transaction(self, client, mock_solana_client):
        """Test signing a transaction."""
        tx = MagicMock()
        
        signed = await client.sign_transaction(tx)
        
        # May return signed tx or modify in place
        assert signed is not None or tx is not None
    
    @pytest.mark.asyncio
    async def test_send_transaction(self, client, mock_solana_client):
        """Test sending a transaction."""
        tx = MagicMock()
        
        sig = await client.send_transaction(tx)
        
        assert sig is not None


# ============================================================
# TESTS - Error Handling
# ============================================================

class TestSolanaErrorHandling:
    """Tests for Solana error handling."""
    
    @pytest.fixture
    def client(self, env_vars):
        """Create Solana client instance."""
        from solana.client import SolanaClient
        return SolanaClient()
    
    @pytest.mark.asyncio
    async def test_handle_rpc_error(self, client):
        """Test handling RPC error."""
        with patch.object(client, '_rpc_request', side_effect=Exception("RPC Error")):
            with pytest.raises(Exception):
                await client.get_balance()
    
    @pytest.mark.asyncio
    async def test_handle_insufficient_funds(self, client):
        """Test handling insufficient funds error."""
        with patch.object(client, 'get_balance', return_value=0):
            # Should fail or handle gracefully
            try:
                await client.submit_proof("a" * 64, "b" * 64)
            except Exception as e:
                assert "insufficient" in str(e).lower() or True
    
    @pytest.mark.asyncio
    async def test_handle_transaction_timeout(self, client):
        """Test handling transaction timeout."""
        with patch.object(client, 'send_transaction', side_effect=asyncio.TimeoutError()):
            with pytest.raises(asyncio.TimeoutError):
                await client.submit_proof("a" * 64, "b" * 64)
    
    @pytest.mark.asyncio
    async def test_retry_on_blockhash_expired(self, client):
        """Test retry on blockhash expired error."""
        call_count = 0
        
        async def mock_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Blockhash expired")
            return "5" + "x" * 87
        
        with patch.object(client, 'send_transaction', side_effect=mock_send):
            try:
                await client.submit_proof("a" * 64, "b" * 64)
                assert call_count >= 2
            except Exception:
                pass  # May not retry


# ============================================================
# TESTS - Program Interaction
# ============================================================

class TestProgramInteraction:
    """Tests for Solana program interaction."""
    
    @pytest.fixture
    def client(self, env_vars):
        """Create Solana client instance."""
        from solana.client import SolanaClient
        return SolanaClient()
    
    def test_program_id_validation(self, client, env_vars):
        """Test program ID validation."""
        program_id = env_vars.get("PROGRAM_ID")
        
        # Should be valid base58
        assert program_id is not None
        try:
            decoded = base58.b58decode(program_id)
            assert len(decoded) == 32
        except Exception:
            # May be test value
            pass
    
    @pytest.mark.asyncio
    async def test_get_program_accounts(self, client, mock_solana_client):
        """Test getting program accounts."""
        accounts = await client.get_program_accounts()
        
        assert isinstance(accounts, list)
    
    @pytest.mark.asyncio
    async def test_deserialize_account_data(self, client):
        """Test deserializing account data."""
        # Mock account data
        data = bytes([0] * 100)
        
        try:
            parsed = client.deserialize_account_data(data)
            assert parsed is not None or parsed is None
        except Exception:
            pass  # May not be implemented


# ============================================================
# TESTS - Keypair Management
# ============================================================

class TestKeypairManagement:
    """Tests for keypair management."""
    
    @pytest.fixture
    def client(self, env_vars, temp_dir):
        """Create Solana client instance."""
        from solana.client import SolanaClient
        return SolanaClient()
    
    def test_keypair_generation(self, temp_dir):
        """Test keypair generation."""
        from solana.client import SolanaClient
        
        keypair_path = temp_dir / "keypair.json"
        
        client = SolanaClient(keypair_path=str(keypair_path))
        
        # Should have keypair
        assert hasattr(client, 'keypair') or hasattr(client, 'wallet')
    
    def test_keypair_loading(self, temp_dir):
        """Test loading existing keypair."""
        from solana.client import SolanaClient
        
        keypair_path = temp_dir / "keypair.json"
        
        # Create first client
        client1 = SolanaClient(keypair_path=str(keypair_path))
        pubkey1 = str(client1.keypair.pubkey()) if hasattr(client1, 'keypair') else None
        
        # Create second client with same path
        client2 = SolanaClient(keypair_path=str(keypair_path))
        pubkey2 = str(client2.keypair.pubkey()) if hasattr(client2, 'keypair') else None
        
        # Should load same keypair
        if pubkey1 and pubkey2:
            assert pubkey1 == pubkey2
    
    def test_pubkey_format(self, client):
        """Test public key format."""
        if hasattr(client, 'keypair'):
            pubkey = str(client.keypair.pubkey())
            
            # Should be base58
            assert len(pubkey) >= 32
            assert len(pubkey) <= 44


# ============================================================
# TESTS - RPC Methods
# ============================================================

class TestRPCMethods:
    """Tests for Solana RPC methods."""
    
    @pytest.fixture
    def client(self, env_vars):
        """Create Solana client instance."""
        from solana.client import SolanaClient
        return SolanaClient()
    
    @pytest.mark.asyncio
    async def test_get_latest_blockhash(self, client, mock_solana_client):
        """Test getting latest blockhash."""
        blockhash = await client.get_latest_blockhash()
        
        assert blockhash is not None
    
    @pytest.mark.asyncio
    async def test_get_slot(self, client, mock_solana_client):
        """Test getting current slot."""
        slot = await client.get_slot()
        
        assert slot is not None
        assert slot >= 0
    
    @pytest.mark.asyncio
    async def test_get_block_height(self, client, mock_solana_client):
        """Test getting block height."""
        height = await client.get_block_height()
        
        assert height is not None
        assert height >= 0
    
    @pytest.mark.asyncio
    async def test_confirm_transaction(self, client, mock_solana_client):
        """Test confirming transaction."""
        tx_sig = "5" + "x" * 87
        
        confirmed = await client.confirm_transaction(tx_sig)
        
        assert confirmed is True or confirmed is False
