"""
Hyperledger Fabric client for D-IBFV vault management.

Uses the `peer` CLI via subprocess to interact with the Fabric test-network.
This approach captures real end-to-end network latency including:
  - TLS handshake
  - Transaction proposal to endorsing peers
  - Ordering service submission
  - Ledger commit

Compatible with fabric-samples test-network (Fabric 2.5.x).
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict


class FabricError(Exception):
    """Raised on Fabric transaction failures."""
    pass


class FabricClient:
    """
    Client for Hyperledger Fabric using the peer CLI.

    Implements the same interface as EVMClient:
        store_vault(uid, cids, t, n) -> dict
        lookup_vault(uid) -> (cids, t, n)
        revoke_vault(uid) -> dict
        vault_status(uid) -> (is_active, is_revoked)
    """

    def __init__(self, connection_info_path: Optional[str] = None):
        """
        Initialize FabricClient.

        Args:
            connection_info_path: Path to connection_info.json from fabric_setup.sh.
                                 If None, auto-detects from the standard location.
        """
        if connection_info_path is None:
            connection_info_path = str(
                Path(__file__).resolve().parent / "blockchain" / "fabric" / "connection_info.json"
            )

        if not os.path.exists(connection_info_path):
            raise FabricError(
                f"Fabric connection info not found: {connection_info_path}\n"
                "Run 'bash fabric_setup.sh' first."
            )

        with open(connection_info_path) as f:
            self.config = json.load(f)

        self.channel = self.config["channel"]
        self.chaincode = self.config["chaincode"]
        self.samples_dir = self.config["fabric_samples_dir"]

        # Set up environment for peer CLI
        self._env = os.environ.copy()
        self._env.update({
            "PATH": f"{self.samples_dir}/bin:{self._env.get('PATH', '')}",
            "FABRIC_CFG_PATH": f"{self.samples_dir}/config",
            "CORE_PEER_TLS_ENABLED": "true",
            "CORE_PEER_LOCALMSPID": "Org1MSP",
            "CORE_PEER_TLS_ROOTCERT_FILE": self.config["tls_cert_org1"],
            "CORE_PEER_MSPCONFIGPATH": self.config["msp_org1"],
            "CORE_PEER_ADDRESS": self.config["peer_org1"],
        })

        # Also need orderer TLS cert for invoke
        orderer_tls = (
            f"{self.samples_dir}/test-network/organizations/"
            "ordererOrganizations/example.com/orderers/orderer.example.com/"
            "msp/tlscacerts/tlsca.example.com-cert.pem"
        )
        self._orderer_tls = orderer_tls
        self._orderer_addr = f"{self.config['orderer']}:7050" if ':' not in self.config['orderer'] else self.config['orderer']

    def _invoke(self, function: str, args: List[str]) -> Tuple[str, float]:
        """
        Invoke a chaincode transaction (write operation).

        Returns (output, latency_ms).
        """
        args_json = json.dumps([{"function": function, "Args": args}])
        # Build ctor JSON
        ctor = json.dumps({"function": function, "Args": args})

        cmd = [
            "peer", "chaincode", "invoke",
            "-o", self._orderer_addr,
            "--ordererTLSHostnameOverride", "orderer.example.com",
            "--tls",
            "--cafile", self._orderer_tls,
            "-C", self.channel,
            "-n", self.chaincode,
            "--peerAddresses", self.config["peer_org1"],
            "--tlsRootCertFiles", self.config["tls_cert_org1"],
            "--peerAddresses", self.config["peer_org2"],
            "--tlsRootCertFiles", (
                f"{self.samples_dir}/test-network/organizations/"
                "peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
            ),
            "-c", ctor,
            "--waitForEvent",
        ]

        t0 = time.perf_counter()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=self._env, timeout=60
        )
        latency = (time.perf_counter() - t0) * 1000

        if result.returncode != 0:
            raise FabricError(
                f"Chaincode invoke failed: {result.stderr.strip()}"
            )

        return result.stdout.strip(), latency

    def _query(self, function: str, args: List[str]) -> Tuple[str, float]:
        """
        Query chaincode (read-only operation, no ordering).

        Returns (output, latency_ms).
        """
        ctor = json.dumps({"function": function, "Args": args})

        cmd = [
            "peer", "chaincode", "query",
            "-C", self.channel,
            "-n", self.chaincode,
            "-c", ctor,
        ]

        t0 = time.perf_counter()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=self._env, timeout=30
        )
        latency = (time.perf_counter() - t0) * 1000

        if result.returncode != 0:
            raise FabricError(
                f"Chaincode query failed: {result.stderr.strip()}"
            )

        return result.stdout.strip(), latency

    def store_vault(self, user_id: str, cids: List[str], t: int, n: int) -> dict:
        """
        Store vault CIDs on the Fabric ledger.

        Returns dict with: tx_hash (N/A), gas_used (N/A), latency_ms.
        """
        cids_json = json.dumps(cids)
        output, latency = self._invoke(
            "StoreVaultCIDs",
            [user_id, cids_json, str(t), str(n)]
        )
        return {
            'tx_hash': 'fabric_tx',  # Fabric doesn't expose tx hash easily via CLI
            'gas_used': 0,  # No gas concept in Fabric
            'latency_ms': latency,
        }

    def lookup_vault(self, user_id: str) -> Tuple[List[str], int, int]:
        """
        Look up vault CIDs from the Fabric ledger.

        Returns (cids, threshold, share_count).
        """
        output, _ = self._query("LookupVaultCIDs", [user_id])

        # Parse the JSON response from chaincode
        try:
            record = json.loads(output)
            return (
                record["cids"],
                record["threshold"],
                record["shareCount"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise FabricError(f"Failed to parse lookup response: {output} - {e}")

    def revoke_vault(self, user_id: str) -> dict:
        """Revoke a vault on the Fabric ledger."""
        output, latency = self._invoke("RevokeVault", [user_id])
        return {
            'tx_hash': 'fabric_tx',
            'gas_used': 0,
            'latency_ms': latency,
        }

    def vault_status(self, user_id: str) -> Tuple[bool, bool]:
        """Returns (is_active, is_revoked)."""
        output, _ = self._query("VaultStatus", [user_id])

        try:
            status = json.loads(output)
            exists = status.get("exists", False)
            revoked = status.get("revoked", False)
            return (exists and not revoked, revoked)
        except (json.JSONDecodeError, KeyError) as e:
            raise FabricError(f"Failed to parse status: {output} - {e}")

    def deploy(self, bytecode=None) -> str:
        """
        No-op for Fabric — chaincode is deployed via fabric_setup.sh.
        Returns the chaincode name.
        """
        return self.chaincode

    @property
    def sender(self) -> str:
        return "Admin@org1.example.com"
