"""
Cryptographic Signing Module for Proof-of-Work Agent.
Handles Ed25519 key generation, signing, and verification.
No external Solana CLI dependencies required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from agent.logger import get_logger


# ============================================================
# Pure Python Ed25519 Implementation
# ============================================================

class Ed25519:
    """
    Pure Python Ed25519 implementation.
    Follows RFC 8032 for signature generation.
    """
    
    # Ed25519 curve parameters
    q = 2**255 - 19
    l = 2**252 + 27742317777372353535851937790883648493
    d = -121665 * pow(121666, q - 2, q) % q
    I = pow(2, (q - 1) // 4, q)
    
    # Base point B
    By = 4 * pow(5, q - 2, q) % q
    Bx = _xrecover(By)
    B = (Bx % q, By % q, 1, Bx * By % q)
    
    @staticmethod
    def _xrecover(y: int) -> int:
        """Recover x from y coordinate."""
        q = Ed25519.q
        xx = (y * y - 1) * pow(Ed25519.d * y * y + 1, q - 2, q)
        x = pow(xx, (q + 3) // 8, q)
        if (x * x - xx) % q != 0:
            x = (x * Ed25519.I) % q
        if x % 2 != 0:
            x = q - x
        return x
    
    @staticmethod
    def _clamp(k: bytes) -> int:
        """Clamp private key scalar."""
        k_list = list(k)
        k_list[0] &= 248
        k_list[31] &= 127
        k_list[31] |= 64
        return int.from_bytes(bytes(k_list), 'little')
    
    @staticmethod
    def _point_add(P: tuple, Q: tuple) -> tuple:
        """Add two points on the curve."""
        q = Ed25519.q
        d = Ed25519.d
        
        x1, y1, z1, t1 = P
        x2, y2, z2, t2 = Q
        
        A = (y1 - x1) * (y2 - x2) % q
        B = (y1 + x1) * (y2 + x2) % q
        C = 2 * t1 * t2 * d % q
        D = 2 * z1 * z2 % q
        E = B - A
        F = D - C
        G = D + C
        H = B + A
        X3 = E * F
        Y3 = G * H
        T3 = E * H
        Z3 = F * G
        
        return (X3 % q, Y3 % q, Z3 % q, T3 % q)
    
    @staticmethod
    def _point_double(P: tuple) -> tuple:
        """Double a point on the curve."""
        q = Ed25519.q
        
        x1, y1, z1, _ = P
        
        A = x1 * x1 % q
        B = y1 * y1 % q
        C = 2 * z1 * z1 % q
        H = A + B
        E = H - (x1 + y1) ** 2 % q
        G = A - B
        F = C + G
        X3 = E * F
        Y3 = G * H
        T3 = E * H
        Z3 = F * G
        
        return (X3 % q, Y3 % q, Z3 % q, T3 % q)
    
    @staticmethod
    def _scalar_mult(P: tuple, e: int) -> tuple:
        """Scalar multiplication (P * e)."""
        if e == 0:
            return (0, 1, 1, 0)
        Q = Ed25519._scalar_mult(P, e // 2)
        Q = Ed25519._point_double(Q)
        if e & 1:
            Q = Ed25519._point_add(Q, P)
        return Q
    
    @staticmethod
    def _point_compress(P: tuple) -> bytes:
        """Compress a point to bytes."""
        q = Ed25519.q
        x, y, z, _ = P
        zi = pow(z, q - 2, q)
        x = (x * zi) % q
        y = (y * zi) % q
        return (y + ((x & 1) << 255)).to_bytes(32, 'little')
    
    @staticmethod
    def _hash_msg(prefix: bytes, msg: bytes) -> int:
        """Hash message with prefix."""
        h = hashlib.sha512(prefix + msg).digest()
        return int.from_bytes(h, 'little')
    
    @classmethod
    def generate_keypair(cls, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Generate Ed25519 keypair.
        
        Returns:
            Tuple of (private_key, public_key) as 32-byte sequences
        """
        if seed is None:
            seed = secrets.token_bytes(32)
        elif len(seed) != 32:
            seed = hashlib.sha256(seed).digest()
        
        # Hash seed to get private key scalar
        h = hashlib.sha512(seed).digest()
        a = cls._clamp(h[:32])
        
        # Compute public key
        A = cls._scalar_mult(cls.B, a)
        public_key = cls._point_compress(A)
        
        return seed, public_key
    
    @classmethod
    def sign(cls, private_key: bytes, message: bytes) -> bytes:
        """
        Sign a message.
        
        Args:
            private_key: 32-byte private key (seed)
            message: Message bytes to sign
            
        Returns:
            64-byte signature
        """
        h = hashlib.sha512(private_key).digest()
        a = cls._clamp(h[:32])
        
        # Hash prefix + message to get r
        r = cls._hash_msg(h[32:], message) % cls.l
        
        # R = r * B
        R = cls._scalar_mult(cls.B, r)
        R_bytes = cls._point_compress(R)
        
        # Get public key
        A = cls._scalar_mult(cls.B, a)
        A_bytes = cls._point_compress(A)
        
        # k = H(R || A || M)
        k = cls._hash_msg(R_bytes + A_bytes, message) % cls.l
        
        # S = (r + k * a) mod l
        S = (r + k * a) % cls.l
        S_bytes = S.to_bytes(32, 'little')
        
        return R_bytes + S_bytes
    
    @classmethod
    def verify(cls, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """
        Verify a signature.
        
        Args:
            public_key: 32-byte public key
            message: Original message bytes
            signature: 64-byte signature
            
        Returns:
            True if valid, False otherwise
        """
        if len(signature) != 64:
            return False
        if len(public_key) != 32:
            return False
        
        try:
            R_bytes = signature[:32]
            S_bytes = signature[32:]
            S = int.from_bytes(S_bytes, 'little')
            
            if S >= cls.l:
                return False
            
            # Decompress R and A
            R = cls._point_decompress(R_bytes)
            A = cls._point_decompress(public_key)
            
            if R is None or A is None:
                return False
            
            # k = H(R || A || M)
            k = cls._hash_msg(R_bytes + public_key, message) % cls.l
            
            # Verify: S * B = R + k * A
            sB = cls._scalar_mult(cls.B, S)
            kA = cls._scalar_mult(A, k)
            RkA = cls._point_add(R, kA)
            
            # Compare points
            return cls._point_compress(sB) == cls._point_compress(RkA)
            
        except Exception:
            return False
    
    @staticmethod
    def _point_decompress(s: bytes) -> Optional[tuple]:
        """Decompress point from bytes."""
        q = Ed25519.q
        
        y = int.from_bytes(s, 'little')
        sign = y >> 255
        y &= (1 << 255) - 1
        
        if y >= q:
            return None
        
        x2 = (y * y - 1) * pow(Ed25519.d * y * y + 1, q - 2, q) % q
        
        if x2 == 0:
            if sign:
                return None
            return (0, y, 1, 0)
        
        x = pow(x2, (q + 3) // 8, q)
        
        if (x * x - x2) % q != 0:
            x = x * Ed25519.I % q
        
        if (x * x - x2) % q != 0:
            return None
        
        if (x & 1) != sign:
            x = q - x
        
        return (x, y, 1, x * y % q)


# ============================================================
# Solana Address Encoding (Base58)
# ============================================================

class Base58:
    """Base58 encoding for Solana addresses."""
    
    ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    
    @classmethod
    def encode(cls, data: bytes) -> str:
        """Encode bytes to base58 string."""
        n = int.from_bytes(data, 'big')
        result = ""
        
        while n > 0:
            n, r = divmod(n, 58)
            result = cls.ALPHABET[r] + result
        
        # Preserve leading zeros
        for byte in data:
            if byte == 0:
                result = cls.ALPHABET[0] + result
            else:
                break
        
        return result or cls.ALPHABET[0]
    
    @classmethod
    def decode(cls, s: str) -> bytes:
        """Decode base58 string to bytes."""
        n = 0
        for char in s:
            n = n * 58 + cls.ALPHABET.index(char)
        
        result = []
        while n > 0:
            n, r = divmod(n, 256)
            result.insert(0, r)
        
        # Preserve leading zeros
        num_leading = 0
        for char in s:
            if char == cls.ALPHABET[0]:
                num_leading += 1
            else:
                break
        
        return bytes([0] * num_leading + result)


# ============================================================
# Keypair Management
# ============================================================

@dataclass
class SolanaKeypair:
    """Solana keypair with signing capabilities."""
    
    private_key: bytes
    public_key: bytes
    
    @property
    def address(self) -> str:
        """Get base58-encoded public key (Solana address)."""
        return Base58.encode(self.public_key)
    
    @property
    def secret_key(self) -> bytes:
        """Get full 64-byte secret key (private + public)."""
        return self.private_key + self.public_key
    
    def sign(self, message: bytes) -> bytes:
        """Sign a message."""
        return Ed25519.sign(self.private_key, message)
    
    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify a signature."""
        return Ed25519.verify(self.public_key, message, signature)
    
    def to_dict(self) -> Dict[str, str]:
        """Export keypair as dictionary (WARNING: contains private key!)."""
        return {
            "public_key": Base58.encode(self.public_key),
            "private_key": Base58.encode(self.private_key),
            "address": self.address
        }
    
    @classmethod
    def generate(cls) -> "SolanaKeypair":
        """Generate new random keypair."""
        private_key, public_key = Ed25519.generate_keypair()
        return cls(private_key=private_key, public_key=public_key)
    
    @classmethod
    def from_seed(cls, seed: bytes) -> "SolanaKeypair":
        """Generate keypair from 32-byte seed."""
        private_key, public_key = Ed25519.generate_keypair(seed)
        return cls(private_key=private_key, public_key=public_key)
    
    @classmethod
    def from_secret_key(cls, secret_key: bytes) -> "SolanaKeypair":
        """Create keypair from 64-byte secret key."""
        if len(secret_key) != 64:
            raise ValueError("Secret key must be 64 bytes")
        return cls(private_key=secret_key[:32], public_key=secret_key[32:])
    
    @classmethod
    def from_json(cls, json_array: List[int]) -> "SolanaKeypair":
        """Create keypair from JSON array (Solana CLI format)."""
        secret_key = bytes(json_array)
        return cls.from_secret_key(secret_key)
    
    def to_json_array(self) -> List[int]:
        """Export as JSON array (Solana CLI compatible format)."""
        return list(self.secret_key)
    
    def save(self, path: Path, password: Optional[str] = None) -> None:
        """Save keypair to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if password:
            # Encrypt with password (simple XOR + hash - use proper encryption in production)
            key = hashlib.sha256(password.encode()).digest()
            encrypted = bytes(a ^ b for a, b in zip(self.secret_key, key * 2))
            data = {
                "encrypted": True,
                "data": base64.b64encode(encrypted).decode()
            }
        else:
            data = self.to_json_array()
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path, password: Optional[str] = None) -> "SolanaKeypair":
        """Load keypair from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and data.get("encrypted"):
            if not password:
                raise ValueError("Password required for encrypted keypair")
            key = hashlib.sha256(password.encode()).digest()
            encrypted = base64.b64decode(data["data"])
            secret_key = bytes(a ^ b for a, b in zip(encrypted, key * 2))
        else:
            secret_key = bytes(data)
        
        return cls.from_secret_key(secret_key)


# ============================================================
# Proof-of-Work Signing
# ============================================================

@dataclass
class SignedProof:
    """Signed proof-of-work result."""
    
    task_id: int
    result_hash: str
    signature: str
    public_key: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    nonce: str = field(default_factory=lambda: secrets.token_hex(8))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "result_hash": self.result_hash,
            "signature": self.signature,
            "public_key": self.public_key,
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }
    
    @property
    def message_bytes(self) -> bytes:
        """Get the message that was signed."""
        msg = f"{self.task_id}:{self.result_hash}:{self.timestamp}:{self.nonce}"
        return msg.encode('utf-8')
    
    def verify(self) -> bool:
        """Verify the signature is valid."""
        try:
            public_key = Base58.decode(self.public_key)
            signature = base64.b64decode(self.signature)
            return Ed25519.verify(public_key, self.message_bytes, signature)
        except Exception:
            return False


class ProofSigner:
    """Signs proof-of-work results with Ed25519."""
    
    MODULE = "signer"
    
    def __init__(self, keypair: Optional[SolanaKeypair] = None, key_path: Optional[Path] = None):
        self.log = get_logger(self.MODULE)
        
        if keypair:
            self.keypair = keypair
        elif key_path and Path(key_path).exists():
            self.keypair = SolanaKeypair.load(key_path)
            self.log.info(f"Loaded keypair from {key_path}")
        else:
            self.keypair = SolanaKeypair.generate()
            self.log.info(f"Generated new keypair: {self.keypair.address}")
            
            # Auto-save if path provided
            if key_path:
                self.keypair.save(key_path)
                self.log.info(f"Saved keypair to {key_path}")
    
    @property
    def address(self) -> str:
        """Get wallet address."""
        return self.keypair.address
    
    @property
    def public_key(self) -> str:
        """Get base58-encoded public key."""
        return self.keypair.address
    
    def sign_proof(self, task_id: int, result_hash: str) -> SignedProof:
        """
        Sign a proof-of-work result.
        
        Args:
            task_id: Task identifier
            result_hash: SHA256 hash of the result
            
        Returns:
            SignedProof with signature
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        nonce = secrets.token_hex(8)
        
        # Create message to sign
        message = f"{task_id}:{result_hash}:{timestamp}:{nonce}"
        message_bytes = message.encode('utf-8')
        
        # Sign
        signature = self.keypair.sign(message_bytes)
        signature_b64 = base64.b64encode(signature).decode('ascii')
        
        proof = SignedProof(
            task_id=task_id,
            result_hash=result_hash,
            signature=signature_b64,
            public_key=self.keypair.address,
            timestamp=timestamp,
            nonce=nonce
        )
        
        self.log.debug(f"Signed proof for task {task_id}")
        return proof
    
    def verify_proof(self, proof: SignedProof) -> bool:
        """Verify a signed proof."""
        return proof.verify()
    
    def sign_message(self, message: Union[str, bytes]) -> str:
        """
        Sign an arbitrary message.
        
        Returns:
            Base64-encoded signature
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        signature = self.keypair.sign(message)
        return base64.b64encode(signature).decode('ascii')
    
    def verify_message(self, message: Union[str, bytes], signature: str, public_key: Optional[str] = None) -> bool:
        """
        Verify a message signature.
        
        Args:
            message: Original message
            signature: Base64-encoded signature
            public_key: Optional public key (uses own if not provided)
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        try:
            sig_bytes = base64.b64decode(signature)
            
            if public_key:
                pk_bytes = Base58.decode(public_key)
            else:
                pk_bytes = self.keypair.public_key
            
            return Ed25519.verify(pk_bytes, message, sig_bytes)
        except Exception as e:
            self.log.warn(f"Signature verification failed: {e}")
            return False


# ============================================================
# Hashchain for Proof History
# ============================================================

@dataclass
class HashchainBlock:
    """Block in the proof hashchain."""
    
    index: int
    timestamp: str
    task_id: int
    result_hash: str
    signature: str
    previous_hash: str
    block_hash: str = ""
    
    def compute_hash(self) -> str:
        """Compute block hash."""
        data = f"{self.index}:{self.timestamp}:{self.task_id}:{self.result_hash}:{self.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "result_hash": self.result_hash,
            "signature": self.signature,
            "previous_hash": self.previous_hash,
            "block_hash": self.block_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HashchainBlock":
        return cls(**data)


class ProofHashchain:
    """
    Hashchain to maintain tamper-evident history of proofs.
    Each block links to the previous via hash.
    """
    
    MODULE = "hashchain"
    
    def __init__(self, chain_file: Optional[Path] = None):
        self.log = get_logger(self.MODULE)
        self.chain_file = chain_file or Path("data/hashchain.json")
        self.chain: List[HashchainBlock] = []
        
        self._load()
    
    def _load(self) -> None:
        """Load chain from file."""
        if self.chain_file.exists():
            try:
                with open(self.chain_file, 'r') as f:
                    data = json.load(f)
                self.chain = [HashchainBlock.from_dict(b) for b in data]
                self.log.info(f"Loaded hashchain with {len(self.chain)} blocks")
            except Exception as e:
                self.log.error(f"Failed to load hashchain: {e}")
                self.chain = []
    
    def _save(self) -> None:
        """Save chain to file."""
        self.chain_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chain_file, 'w') as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=2)
    
    @property
    def last_hash(self) -> str:
        """Get hash of last block."""
        if not self.chain:
            return "0" * 64  # Genesis hash
        return self.chain[-1].block_hash
    
    def add_proof(self, signed_proof: SignedProof) -> HashchainBlock:
        """Add a new proof to the chain."""
        block = HashchainBlock(
            index=len(self.chain),
            timestamp=signed_proof.timestamp,
            task_id=signed_proof.task_id,
            result_hash=signed_proof.result_hash,
            signature=signed_proof.signature,
            previous_hash=self.last_hash
        )
        block.block_hash = block.compute_hash()
        
        self.chain.append(block)
        self._save()
        
        self.log.info(f"Added block #{block.index} to hashchain")
        return block
    
    def verify_chain(self) -> bool:
        """Verify entire chain integrity."""
        if not self.chain:
            return True
        
        # Check genesis block
        if self.chain[0].previous_hash != "0" * 64:
            self.log.error("Invalid genesis block")
            return False
        
        # Check each block
        for i, block in enumerate(self.chain):
            # Verify block hash
            if block.block_hash != block.compute_hash():
                self.log.error(f"Block {i} hash mismatch")
                return False
            
            # Verify chain link
            if i > 0 and block.previous_hash != self.chain[i-1].block_hash:
                self.log.error(f"Block {i} chain link broken")
                return False
        
        self.log.info(f"Hashchain verified: {len(self.chain)} blocks valid")
        return True
    
    def get_proof_by_task(self, task_id: int) -> Optional[HashchainBlock]:
        """Find proof for a specific task."""
        for block in reversed(self.chain):
            if block.task_id == task_id:
                return block
        return None
    
    def get_recent_proofs(self, count: int = 10) -> List[HashchainBlock]:
        """Get most recent proofs."""
        return self.chain[-count:]


# ============================================================
# Convenience Functions
# ============================================================

_default_signer: Optional[ProofSigner] = None


def get_signer() -> ProofSigner:
    """Get or create the default proof signer."""
    global _default_signer
    if _default_signer is None:
        key_path = Path(os.getenv("SOLANA_KEYPAIR_PATH", "data/keypair.json"))
        _default_signer = ProofSigner(key_path=key_path)
    return _default_signer


def sign_result(task_id: int, result: str) -> SignedProof:
    """Sign a task result and return proof."""
    result_hash = hashlib.sha256(result.encode()).hexdigest()
    return get_signer().sign_proof(task_id, result_hash)


def get_wallet_address() -> str:
    """Get the agent's wallet address."""
    return get_signer().address
