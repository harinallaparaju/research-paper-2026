"""
Unified blockchain client for D-IBFV vault CID management.

Supports three backends:
  - GanacheClient  — local Ganache EVM (for development/benchmarking)
  - EthereumClient — Ethereum L1 (Sepolia testnet or mainnet)
  - PolygonClient  — Polygon PoS (Amoy testnet or mainnet)

All clients implement the same interface:
  store_vault(uid, cids, t, n) -> tx_hash
  lookup_vault(uid) -> (cids, t, n)
  revoke_vault(uid) -> tx_hash
  vault_status(uid) -> (is_active, is_revoked)

The Solidity contract ABI is embedded — no external compilation needed
for deployment. The contract bytecode is compiled once and cached.
"""

import json
import time
import hashlib
import subprocess
import os
import threading
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware


# ─── Contract ABI (matches VaultRegistry.sol) ───

VAULT_REGISTRY_ABI = [
    {
        "inputs": [
            {"name": "uid", "type": "bytes32"},
            {"name": "cids", "type": "string[]"},
            {"name": "t", "type": "uint8"},
            {"name": "n", "type": "uint8"}
        ],
        "name": "storeVaultCIDs",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "uid", "type": "bytes32"}],
        "name": "lookupVaultCIDs",
        "outputs": [
            {"name": "cids", "type": "string[]"},
            {"name": "t", "type": "uint8"},
            {"name": "n", "type": "uint8"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "uid", "type": "bytes32"}],
        "name": "revokeVault",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "uid", "type": "bytes32"}],
        "name": "vaultStatus",
        "outputs": [
            {"name": "isActive", "type": "bool"},
            {"name": "isRevoked", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "uid", "type": "bytes32"},
            {"name": "threshold", "type": "uint8"},
            {"name": "shareCount", "type": "uint8"},
            {"name": "timestamp", "type": "uint256"}
        ],
        "name": "VaultStored",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "uid", "type": "bytes32"},
            {"name": "timestamp", "type": "uint256"}
        ],
        "name": "VaultRevoked",
        "type": "event"
    }
]


def _uid_from_string(user_id: str) -> bytes:
    """Convert a string user ID to bytes32 via keccak256."""
    return Web3.solidity_keccak(['string'], [user_id])


def _compile_contract() -> str:
    """
    Compile VaultRegistry.sol and return bytecode.
    Uses solcjs if available, otherwise falls back to the Docker solc image.
    """
    sol_path = Path(__file__).parent / "blockchain" / "ethereum" / "VaultRegistry.sol"
    if not sol_path.exists():
        raise FileNotFoundError(f"Contract not found: {sol_path}")

    # Try solc via Docker (OrbStack)
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "-v",
             f"{sol_path.parent}:/sources",
             "ethereum/solc:0.8.20",
             "--combined-json", "bin",
             "/sources/VaultRegistry.sol"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for key, val in data["contracts"].items():
                if "VaultRegistry" in key:
                    return "0x" + val["bin"]
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback: try py-solc-x
    try:
        from solcx import compile_source, install_solc
        install_solc("0.8.20")
        compiled = compile_source(
            sol_path.read_text(),
            output_values=["bin"],
            solc_version="0.8.20"
        )
        for key, val in compiled.items():
            if "VaultRegistry" in key:
                return val["bin"]
    except ImportError:
        pass

    raise RuntimeError(
        "Cannot compile Solidity. Install Docker (OrbStack) with ethereum/solc:0.8.20 "
        "or pip install py-solc-x"
    )


class EVMClient:
    """Base class for EVM-compatible blockchain clients."""

    def __init__(self, rpc_url: str, private_key: Optional[str] = None,
                 contract_address: Optional[str] = None,
                 chain_id: int = 1337, poa: bool = False):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if poa:
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self.chain_id = chain_id

        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
        else:
            # Use first account (Ganache default)
            self.account = None

        self.contract_address = contract_address
        self.contract = None
        self._nonce_lock = threading.Lock()
        if contract_address:
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(contract_address),
                abi=VAULT_REGISTRY_ABI
            )

    @property
    def sender(self) -> str:
        if self.account:
            return self.account.address
        return self.w3.eth.accounts[0]

    def deploy(self, bytecode: Optional[str] = None) -> str:
        """Deploy VaultRegistry contract. Returns contract address."""
        if bytecode is None:
            bytecode = _compile_contract()

        contract = self.w3.eth.contract(abi=VAULT_REGISTRY_ABI, bytecode=bytecode)

        if self.account:
            tx = contract.constructor().build_transaction({
                'from': self.sender,
                'nonce': self.w3.eth.get_transaction_count(self.sender),
                'gas': 3_000_000,
                'gasPrice': self.w3.eth.gas_price,
                'chainId': self.chain_id,
            })
            signed = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        else:
            tx_hash = contract.constructor().transact({'from': self.sender, 'gas': 3_000_000})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"Deploy transaction reverted (status={receipt.status})")
        self.contract_address = receipt.contractAddress
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=VAULT_REGISTRY_ABI
        )
        return self.contract_address

    def _send_tx(self, fn):
        """Build, sign, send and wait for a contract function call (thread-safe)."""
        with self._nonce_lock:
            if self.account:
                tx = fn.build_transaction({
                    'from': self.sender,
                    'nonce': self.w3.eth.get_transaction_count(self.sender),
                    'gas': 3_000_000,
                    'gasPrice': self.w3.eth.gas_price,
                    'chainId': self.chain_id,
                })
                signed = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            else:
                tx_hash = fn.transact({'from': self.sender, 'gas': 3_000_000})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(
                f"Transaction reverted (status={receipt.status}, gas={receipt.gasUsed})"
            )
        return receipt

    def store_vault(self, user_id: str, cids: List[str], t: int, n: int) -> dict:
        """
        Store vault CIDs on-chain.

        Returns dict with: tx_hash, gas_used, block_number, latency_ms.
        """
        uid = _uid_from_string(user_id)
        fn = self.contract.functions.storeVaultCIDs(uid, cids, t, n)
        t0 = time.perf_counter()
        receipt = self._send_tx(fn)
        latency = (time.perf_counter() - t0) * 1000
        return {
            'tx_hash': receipt.transactionHash.hex(),
            'gas_used': receipt.gasUsed,
            'block_number': receipt.blockNumber,
            'latency_ms': latency,
        }

    def lookup_vault(self, user_id: str) -> Tuple[List[str], int, int]:
        """
        Look up vault CIDs from on-chain.

        Returns (cids, threshold, share_count).
        """
        uid = _uid_from_string(user_id)
        return self.contract.functions.lookupVaultCIDs(uid).call()

    def revoke_vault(self, user_id: str) -> dict:
        """Revoke a vault. Returns tx receipt info."""
        uid = _uid_from_string(user_id)
        fn = self.contract.functions.revokeVault(uid)
        t0 = time.perf_counter()
        receipt = self._send_tx(fn)
        latency = (time.perf_counter() - t0) * 1000
        return {
            'tx_hash': receipt.transactionHash.hex(),
            'gas_used': receipt.gasUsed,
            'latency_ms': latency,
        }

    def vault_status(self, user_id: str) -> Tuple[bool, bool]:
        """Returns (is_active, is_revoked)."""
        uid = _uid_from_string(user_id)
        return self.contract.functions.vaultStatus(uid).call()


class GanacheClient(EVMClient):
    """Client for local Ganache EVM (development/benchmarking)."""

    DOCKER_IMAGE = "trufflesuite/ganache:latest"
    CONTAINER_NAME = "dibfv-ganache"

    def __init__(self, rpc_url: str = "http://127.0.0.1:8545",
                 contract_address: Optional[str] = None):
        super().__init__(rpc_url, chain_id=1337, contract_address=contract_address)

    @classmethod
    def start_ganache(cls, port: int = 8545, accounts: int = 10,
                      block_time: Optional[int] = None) -> 'GanacheClient':
        """
        Start a Ganache container via Docker (OrbStack) and return a connected client.
        """
        # Stop any existing container
        subprocess.run(
            ["docker", "rm", "-f", cls.CONTAINER_NAME],
            capture_output=True, timeout=10
        )

        cmd = [
            "docker", "run", "-d",
            "--name", cls.CONTAINER_NAME,
            "-p", f"{port}:8545",
            cls.DOCKER_IMAGE,
            f"--accounts={accounts}",
            "--deterministic",
            "--gasLimit=12000000",
        ]
        if block_time is not None:
            cmd.append(f"--blockTime={block_time}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Ganache: {result.stderr}")

        # Wait for Ganache to be ready
        rpc_url = f"http://127.0.0.1:{port}"
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        for _ in range(30):
            try:
                if w3.is_connected():
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Ganache failed to start within 15s")

        return cls(rpc_url=rpc_url)

    @classmethod
    def stop_ganache(cls):
        """Stop and remove the Ganache container."""
        subprocess.run(
            ["docker", "rm", "-f", cls.CONTAINER_NAME],
            capture_output=True, timeout=10
        )


class EthereumClient(EVMClient):
    """Client for Ethereum L1 (Sepolia testnet)."""

    def __init__(self, rpc_url: str, private_key: str,
                 contract_address: Optional[str] = None):
        super().__init__(rpc_url, private_key, contract_address,
                         chain_id=11155111)  # Sepolia


class PolygonClient(EVMClient):
    """Client for Polygon PoS (Amoy testnet)."""

    def __init__(self, rpc_url: str, private_key: str,
                 contract_address: Optional[str] = None):
        super().__init__(rpc_url, private_key, contract_address,
                         chain_id=80002, poa=True)  # Amoy
