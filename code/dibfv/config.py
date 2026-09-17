"""
D-IBFV Platform Configuration.

Reads credentials from environment variables or .env file.
Never hardcode private keys in source code.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def _load_env():
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, val = line.partition('=')
                os.environ.setdefault(key.strip(), val.strip())


_load_env()


@dataclass
class PlatformConfig:
    """Configuration for a single blockchain platform."""
    name: str
    rpc_url: str
    private_key: Optional[str]
    chain_id: int
    poa: bool = False
    contract_address: Optional[str] = None


# ─── Platform Configs ───

GANACHE = PlatformConfig(
    name="Ganache",
    rpc_url=os.environ.get("GANACHE_RPC", "http://127.0.0.1:8545"),
    private_key=None,  # Uses first Ganache account
    chain_id=1337,
)

SEPOLIA = PlatformConfig(
    name="Ethereum Sepolia",
    rpc_url=os.environ.get("SEPOLIA_RPC", "https://rpc.sepolia.org"),
    private_key=os.environ.get("TESTNET_PRIVATE_KEY"),
    chain_id=11155111,
)

AMOY = PlatformConfig(
    name="Polygon Amoy",
    rpc_url=os.environ.get("AMOY_RPC", "https://rpc-amoy.polygon.technology"),
    private_key=os.environ.get("TESTNET_PRIVATE_KEY"),
    chain_id=80002,
    poa=True,
)

FABRIC = PlatformConfig(
    name="Hyperledger Fabric",
    rpc_url=os.environ.get("FABRIC_GATEWAY_URL", "http://127.0.0.1:7054"),
    private_key=None,
    chain_id=0,  # Not EVM
)


def get_all_evm_platforms():
    """Return all EVM platform configs that have credentials."""
    platforms = []
    for cfg in [GANACHE, SEPOLIA, AMOY]:
        if cfg.private_key or cfg.name == "Ganache":
            platforms.append(cfg)
    return platforms


def get_paper_platforms():
    """Return the three platforms referenced in the paper."""
    return [FABRIC, SEPOLIA, AMOY]
