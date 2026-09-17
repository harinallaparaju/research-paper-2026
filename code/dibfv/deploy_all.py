"""
Deploy VaultRegistry contract to all EVM platforms.

Usage:
    python deploy_all.py              # deploy to all configured platforms
    python deploy_all.py --sepolia    # deploy to Sepolia only
    python deploy_all.py --amoy       # deploy to Amoy only
    python deploy_all.py --ganache    # deploy to Ganache only
"""

import sys
import os
import json
import argparse
from pathlib import Path

code_dir = str(Path(__file__).resolve().parent.parent)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from dibfv.config import GANACHE, SEPOLIA, AMOY
from dibfv.blockchain_client import (
    GanacheClient, EthereumClient, PolygonClient, _compile_contract
)

ENV_PATH = Path(__file__).resolve().parent / ".env"
DEPLOY_RECORD = Path(__file__).resolve().parent / "deployed_contracts.json"


def _update_env(key, value):
    """Update .env file with a key=value pair."""
    lines = []
    found = False
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.strip().startswith(key + '='):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text('\n'.join(lines) + '\n')


def deploy_ganache():
    """Deploy to local Ganache."""
    print("\n=== Deploying to Ganache ===")
    client = GanacheClient(rpc_url=GANACHE.rpc_url)
    if not client.w3.is_connected():
        print("  ERROR: Ganache not running. Start with 'docker start dibfv-ganache'")
        return None
    bytecode = _compile_contract()
    addr = client.deploy(bytecode)
    print(f"  Contract: {addr}")
    print(f"  Balance: {client.w3.eth.get_balance(client.sender) / 1e18:.2f} ETH")
    return {"platform": "Ganache", "address": addr, "chain_id": 1337}


def deploy_sepolia():
    """Deploy to Ethereum Sepolia."""
    print("\n=== Deploying to Ethereum Sepolia ===")
    if not SEPOLIA.private_key:
        print("  ERROR: TESTNET_PRIVATE_KEY not set in .env")
        return None

    client = EthereumClient(
        rpc_url=SEPOLIA.rpc_url,
        private_key=SEPOLIA.private_key,
    )

    if not client.w3.is_connected():
        print(f"  ERROR: Cannot connect to {SEPOLIA.rpc_url}")
        return None

    balance = client.w3.eth.get_balance(client.sender)
    print(f"  Address: {client.sender}")
    print(f"  Balance: {balance / 1e18:.6f} ETH")

    if balance < 0.01 * 1e18:
        print(f"  ERROR: Insufficient balance. Need ~0.01 ETH for deployment.")
        print(f"  Fund at: https://cloud.google.com/application/web3/faucet/ethereum/sepolia")
        return None

    bytecode = _compile_contract()
    print("  Deploying (this may take 15-30s for block confirmation)...")
    addr = client.deploy(bytecode)
    print(f"  Contract: {addr}")
    _update_env("SEPOLIA_CONTRACT", addr)
    return {"platform": "Ethereum Sepolia", "address": addr, "chain_id": 11155111}


def deploy_amoy():
    """Deploy to Polygon Amoy."""
    print("\n=== Deploying to Polygon Amoy ===")
    if not AMOY.private_key:
        print("  ERROR: TESTNET_PRIVATE_KEY not set in .env")
        return None

    client = PolygonClient(
        rpc_url=AMOY.rpc_url,
        private_key=AMOY.private_key,
    )

    if not client.w3.is_connected():
        print(f"  ERROR: Cannot connect to {AMOY.rpc_url}")
        return None

    balance = client.w3.eth.get_balance(client.sender)
    print(f"  Address: {client.sender}")
    print(f"  Balance: {balance / 1e18:.6f} MATIC")

    if balance < 0.1 * 1e18:
        print(f"  ERROR: Insufficient balance. Need ~0.1 MATIC for deployment.")
        print(f"  Fund at: https://faucet.polygon.technology/")
        return None

    bytecode = _compile_contract()
    print("  Deploying (this may take 5-15s)...")
    addr = client.deploy(bytecode)
    print(f"  Contract: {addr}")
    _update_env("AMOY_CONTRACT", addr)
    return {"platform": "Polygon Amoy", "address": addr, "chain_id": 80002}


def main():
    parser = argparse.ArgumentParser(description="Deploy VaultRegistry to blockchain platforms")
    parser.add_argument("--ganache", action="store_true", help="Deploy to Ganache only")
    parser.add_argument("--sepolia", action="store_true", help="Deploy to Sepolia only")
    parser.add_argument("--amoy", action="store_true", help="Deploy to Amoy only")
    args = parser.parse_args()

    deploy_all = not (args.ganache or args.sepolia or args.amoy)

    results = []

    if args.ganache or deploy_all:
        r = deploy_ganache()
        if r:
            results.append(r)

    if args.sepolia or deploy_all:
        r = deploy_sepolia()
        if r:
            results.append(r)

    if args.amoy or deploy_all:
        r = deploy_amoy()
        if r:
            results.append(r)

    # Save deployment record
    existing = {}
    if DEPLOY_RECORD.exists():
        existing = json.loads(DEPLOY_RECORD.read_text())

    for r in results:
        existing[r["platform"]] = r

    DEPLOY_RECORD.write_text(json.dumps(existing, indent=2) + '\n')

    print(f"\n{'='*50}")
    print(f"Deployed: {len(results)} platform(s)")
    for r in results:
        print(f"  {r['platform']}: {r['address']}")
    print(f"Record saved to: {DEPLOY_RECORD}")


if __name__ == '__main__':
    main()
