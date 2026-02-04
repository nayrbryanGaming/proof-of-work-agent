from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import requests

from agent.config import config
from agent.logger import get_logger

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ALPHABET_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    res = ""
    while n > 0:
        n, rem = divmod(n, 58)
        res = ALPHABET[rem] + res
    pad = 0
    for b in data:
        if b == 0:
            pad += 1
        else:
            break
    return ("1" * pad) + (res or "")


def b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        if ch not in ALPHABET_INDEX:
            raise ValueError("Invalid base58 string.")
        n = n * 58 + ALPHABET_INDEX[ch]
    full = n.to_bytes((n.bit_length() + 7) // 8, "big") if n > 0 else b""
    pad = 0
    for ch in s:
        if ch == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + full


def shortvec_encode(value: int) -> bytes:
    out = bytearray()
    v = value
    while True:
        elem = v & 0x7F
        v >>= 7
        if v:
            elem |= 0x80
        out.append(elem)
        if not v:
            break
    return bytes(out)


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


def edwards(P: Tuple[int, int], Q: Tuple[int, int]) -> Tuple[int, int]:
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * inv(1 + d * x1 * x2 * y1 * y2) % q
    y3 = (y1 * y2 + x1 * x2) * inv(1 - d * x1 * x2 * y1 * y2) % q
    return (x3, y3)


def scalarmult(P: Tuple[int, int], e: int) -> Tuple[int, int]:
    if e == 0:
        return (0, 1)
    Q = (0, 1)
    N = P
    k = e
    while k > 0:
        if k & 1:
            Q = edwards(Q, N)
        N = edwards(N, N)
        k >>= 1
    return Q


def encodepoint(P: Tuple[int, int]) -> bytes:
    x, y = P
    bits = y | ((x & 1) << 255)
    return bits.to_bytes(32, "little")


def isoncurve(P: Tuple[int, int]) -> bool:
    x, y = P
    return (-x * x + y * y - 1 - d * x * x * y * y) % q == 0


def decodepoint(s: bytes) -> Tuple[int, int]:
    if len(s) != 32:
        raise ValueError("Invalid point length.")
    y = int.from_bytes(s, "little") & ((1 << 255) - 1)
    xsign = s[31] >> 7
    x = xrecover(y)
    if x & 1 != xsign:
        x = q - x
    P = (x, y)
    if not isoncurve(P):
        raise ValueError("Point not on curve.")
    return P


By = (4 * inv(5)) % q
Bx = xrecover(By)
B = (Bx, By)


def _secret_expand(seed: bytes) -> Tuple[int, bytes]:
    h = H(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    prefix = h[32:]
    return a, prefix


def ed25519_publickey(seed: bytes) -> bytes:
    a, _ = _secret_expand(seed)
    A = scalarmult(B, a)
    return encodepoint(A)


def ed25519_sign(message: bytes, seed: bytes, public_key: bytes) -> bytes:
    a, prefix = _secret_expand(seed)
    r = int.from_bytes(H(prefix + message), "little") % l
    R = scalarmult(B, r)
    Renc = encodepoint(R)
    k = int.from_bytes(H(Renc + public_key + message), "little") % l
    S = (r + k * a) % l
    return Renc + S.to_bytes(32, "little")


@dataclass(frozen=True)
class PublicKey:
    data: bytes

    def __bytes__(self) -> bytes:
        return self.data

    def to_base58(self) -> str:
        return b58encode(self.data)

    @classmethod
    def from_base58(cls, s: str) -> "PublicKey":
        data = b58decode(s)
        if len(data) != 32:
            raise ValueError("Public key must be 32 bytes.")
        return cls(data)


@dataclass
class AccountMeta:
    pubkey: PublicKey
    is_signer: bool
    is_writable: bool


@dataclass
class Instruction:
    program_id: PublicKey
    accounts: List[AccountMeta]
    data: bytes


class Keypair:
    def __init__(self, secret: bytes):
        if len(secret) == 64:
            seed = secret[:32]
        elif len(secret) == 32:
            seed = secret
        else:
            raise ValueError("Secret key must be 32 or 64 bytes.")
        self.seed = seed
        self.public_key = PublicKey(ed25519_publickey(seed))

    def sign(self, message: bytes) -> bytes:
        return ed25519_sign(message, self.seed, self.public_key.data)


def _load_keypair(session: str) -> Keypair:
    if not session:
        raise ValueError("AGENTWALLET_SESSION is required.")
    session = session.strip()
    path = Path(session)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return Keypair(bytes(data))
    if session.startswith("["):
        data = json.loads(session)
        if isinstance(data, list):
            return Keypair(bytes(data))
    try:
        secret = b58decode(session)
        if len(secret) in (32, 64):
            return Keypair(secret)
    except Exception:
        pass
    try:
        secret = bytes.fromhex(session)
        if len(secret) in (32, 64):
            return Keypair(secret)
    except Exception:
        pass
    raise ValueError("AGENTWALLET_SESSION must be a Solana keypair string or path.")


def is_on_curve(pubkey_bytes: bytes) -> bool:
    try:
        decodepoint(pubkey_bytes)
        return True
    except Exception:
        return False


def create_program_address(seeds: List[bytes], program_id: PublicKey) -> PublicKey:
    buffer = b"".join(seeds) + bytes(program_id) + b"ProgramDerivedAddress"
    hash_bytes = hashlib.sha256(buffer).digest()
    if is_on_curve(hash_bytes):
        raise ValueError("Invalid seeds, address is on curve.")
    return PublicKey(hash_bytes)


def find_program_address(seeds: List[bytes], program_id: PublicKey) -> Tuple[PublicKey, int]:
    for bump in range(255, -1, -1):
        try:
            addr = create_program_address(seeds + [bytes([bump])], program_id)
            return addr, bump
        except ValueError:
            continue
    raise RuntimeError("Unable to find program address.")


def borsh_u64(value: int) -> bytes:
    return value.to_bytes(8, "little")


def borsh_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return len(data).to_bytes(4, "little") + data


def anchor_discriminator(name: str) -> bytes:
    return hashlib.sha256(f"global:{name}".encode("utf-8")).digest()[:8]


def compile_message(
    instructions: List[Instruction], payer: PublicKey, recent_blockhash: str
) -> bytes:
    meta_map = {}
    order = []

    def add(pubkey: PublicKey, is_signer: bool, is_writable: bool) -> None:
        key = pubkey.to_base58()
        if key in meta_map:
            cur_signer, cur_writable, _ = meta_map[key]
            meta_map[key] = (cur_signer or is_signer, cur_writable or is_writable, pubkey)
        else:
            meta_map[key] = (is_signer, is_writable, pubkey)
            order.append(key)

    add(payer, True, True)
    for ix in instructions:
        for meta in ix.accounts:
            add(meta.pubkey, meta.is_signer, meta.is_writable)
        add(ix.program_id, False, False)

    signer_keys = [k for k in order if meta_map[k][0]]
    nonsigner_keys = [k for k in order if not meta_map[k][0]]
    ordered_keys = signer_keys + nonsigner_keys

    num_required = len(signer_keys)
    num_readonly_signed = sum(1 for k in signer_keys if not meta_map[k][1])
    num_readonly_unsigned = sum(1 for k in nonsigner_keys if not meta_map[k][1])

    header = bytes([num_required, num_readonly_signed, num_readonly_unsigned])
    account_keys = b"".join(bytes(meta_map[k][2]) for k in ordered_keys)
    account_len = shortvec_encode(len(ordered_keys))

    blockhash_bytes = b58decode(recent_blockhash)
    if len(blockhash_bytes) != 32:
        raise ValueError("Invalid blockhash length.")

    index_map = {ordered_keys[i]: i for i in range(len(ordered_keys))}

    ix_data_list = []
    for ix in instructions:
        program_index = index_map[ix.program_id.to_base58()]
        acct_indices = [index_map[meta.pubkey.to_base58()] for meta in ix.accounts]
        ix_data = (
            bytes([program_index])
            + shortvec_encode(len(acct_indices))
            + bytes(acct_indices)
            + shortvec_encode(len(ix.data))
            + ix.data
        )
        ix_data_list.append(ix_data)

    instructions_bytes = shortvec_encode(len(ix_data_list)) + b"".join(ix_data_list)
    return header + account_len + account_keys + blockhash_bytes + instructions_bytes


def serialize_transaction(message: bytes, signatures: List[bytes]) -> bytes:
    return shortvec_encode(len(signatures)) + b"".join(signatures) + message


class SolanaRPC:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()

    def _post(self, method: str, params: list) -> dict:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        resp = self.session.post(self.url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["result"]

    def get_latest_blockhash(self) -> str:
        res = self._post("getLatestBlockhash", [])
        return res["value"]["blockhash"]

    def send_transaction(self, tx_b64: str) -> str:
        return self._post(
            "sendTransaction",
            [tx_b64, {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}],
        )

    def get_account_info(self, pubkey: str) -> dict | None:
        res = self._post("getAccountInfo", [pubkey, {"encoding": "base64"}])
        return res.get("value")

    def get_signature_status(self, signature: str) -> dict | None:
        res = self._post(
            "getSignatureStatuses", [[signature], {"searchTransactionHistory": True}]
        )
        value = res.get("value")
        if not value:
            return None
        return value[0]


SYSTEM_PROGRAM_ID = PublicKey.from_base58("11111111111111111111111111111111")
DEFAULT_DESCRIPTION = "Proof-of-work bounty"
DEFAULT_REWARD = 0


def build_create_bounty_ix(
    program_id: PublicKey,
    bounty_pda: PublicKey,
    authority: PublicKey,
    bounty_id: int,
    description: str,
    reward: int,
) -> Instruction:
    data = (
        anchor_discriminator("create_bounty")
        + borsh_u64(bounty_id)
        + borsh_string(description)
        + borsh_u64(reward)
    )
    accounts = [
        AccountMeta(bounty_pda, False, True),
        AccountMeta(authority, True, True),
        AccountMeta(SYSTEM_PROGRAM_ID, False, False),
    ]
    return Instruction(program_id=program_id, accounts=accounts, data=data)


def build_submit_work_ix(
    program_id: PublicKey,
    bounty_pda: PublicKey,
    authority: PublicKey,
    result_hash: str,
) -> Instruction:
    data = anchor_discriminator("submit_work") + borsh_string(result_hash)
    accounts = [
        AccountMeta(bounty_pda, False, True),
        AccountMeta(authority, True, False),
    ]
    return Instruction(program_id=program_id, accounts=accounts, data=data)


class SolanaClient:
    def __init__(
        self,
        rpc_url: str | None = None,
        program_id: str | None = None,
        session: str | None = None,
    ) -> None:
        self.log = get_logger("solana.client")
        self.rpc = SolanaRPC(rpc_url or config.solana_rpc)
        pid = program_id or config.program_id
        if not pid:
            raise ValueError("PROGRAM_ID is required.")
        self.program_id = PublicKey.from_base58(pid)
        self.keypair = _load_keypair(session or config.agentwallet_session)
        self.bounty_id = int(os.getenv("BOUNTY_ID", "1"))

    def _bounty_pda(self) -> Tuple[PublicKey, int]:
        seeds = [b"bounty", self.bounty_id.to_bytes(8, "little")]
        return find_program_address(seeds, self.program_id)

    def _bounty_exists(self, bounty_pubkey: PublicKey) -> bool:
        info = self.rpc.get_account_info(bounty_pubkey.to_base58())
        return info is not None

    def _send_instructions(self, instructions: List[Instruction]) -> str:
        blockhash = self.rpc.get_latest_blockhash()
        message = compile_message(instructions, self.keypair.public_key, blockhash)
        signature = self.keypair.sign(message)
        tx = serialize_transaction(message, [signature])
        tx_b64 = base64.b64encode(tx).decode("utf-8")
        return self.rpc.send_transaction(tx_b64)

    def _create_bounty(self, bounty_pubkey: PublicKey) -> str:
        ix = build_create_bounty_ix(
            program_id=self.program_id,
            bounty_pda=bounty_pubkey,
            authority=self.keypair.public_key,
            bounty_id=self.bounty_id,
            description=DEFAULT_DESCRIPTION,
            reward=DEFAULT_REWARD,
        )
        return self._send_instructions([ix])

    def _submit_work(self, bounty_pubkey: PublicKey, proof_hash: str) -> str:
        ix = build_submit_work_ix(
            program_id=self.program_id,
            bounty_pda=bounty_pubkey,
            authority=self.keypair.public_key,
            result_hash=proof_hash,
        )
        return self._send_instructions([ix])

    def submit_proof(self, proof_hash: str) -> str:
        bounty_pubkey, _ = self._bounty_pda()
        if not self._bounty_exists(bounty_pubkey):
            self.log.info("Bounty account missing; creating.")
            self._create_bounty(bounty_pubkey)
        sig = self._submit_work(bounty_pubkey, proof_hash)
        self.log.info(f"Submitted proof: {sig}")
        return sig

    def verify_signature(self, signature: str) -> bool:
        status = self.rpc.get_signature_status(signature)
        if not status:
            return False
        if status.get("err") is not None:
            return False
        return status.get("confirmationStatus") in ("confirmed", "finalized")
