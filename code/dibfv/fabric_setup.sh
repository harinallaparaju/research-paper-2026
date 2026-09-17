#!/usr/bin/env bash
# D-IBFV Hyperledger Fabric Test-Network Setup
#
# Prerequisites: Docker Desktop / OrbStack, curl, git
#
# This script:
#   1. Clones fabric-samples (if not present)
#   2. Downloads Fabric binaries + Docker images
#   3. Starts the test-network with CouchDB + CA
#   4. Creates channel 'dibfvchannel'
#   5. Packages, installs, and commits the vault_registry chaincode
#
# Usage: bash fabric_setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FABRIC_DIR="${SCRIPT_DIR}/blockchain/fabric"
SAMPLES_DIR="${FABRIC_DIR}/fabric-samples"
CHAINCODE_DIR="${FABRIC_DIR}/chaincode"
CHANNEL_NAME="dibfvchannel"
CC_NAME="vault_registry"
CC_VERSION="1.0"
CC_SEQUENCE=1

echo "============================================"
echo " D-IBFV Fabric Test-Network Setup"
echo "============================================"

# 1. Clone fabric-samples if needed
if [ ! -d "$SAMPLES_DIR" ]; then
    echo "[1/5] Cloning fabric-samples..."
    cd "$FABRIC_DIR"
    curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh
    chmod +x install-fabric.sh
    ./install-fabric.sh --fabric-version 2.5.9 docker samples binary
    echo "  Done."
else
    echo "[1/5] fabric-samples already present."
fi

export PATH="${SAMPLES_DIR}/bin:$PATH"
export FABRIC_CFG_PATH="${SAMPLES_DIR}/config"

# 2. Start test-network
echo "[2/5] Starting test-network..."
cd "${SAMPLES_DIR}/test-network"

# Bring down any existing network
./network.sh down 2>/dev/null || true

# Start with CouchDB and Certificate Authorities
./network.sh up createChannel -c "$CHANNEL_NAME" -ca -s couchdb
echo "  Network up, channel '$CHANNEL_NAME' created."

# 3. Prepare chaincode
echo "[3/5] Preparing chaincode..."
CC_SRC_PATH="${CHAINCODE_DIR}"

# Ensure go.mod is tidy
cd "$CC_SRC_PATH"
GO111MODULE=on go mod tidy 2>/dev/null || true
cd "${SAMPLES_DIR}/test-network"

# 4. Deploy chaincode
echo "[4/5] Deploying chaincode '${CC_NAME}'..."
./network.sh deployCC \
    -ccn "$CC_NAME" \
    -ccp "$CC_SRC_PATH" \
    -ccl go \
    -c "$CHANNEL_NAME" \
    -ccv "$CC_VERSION" \
    -ccs "$CC_SEQUENCE"

echo "  Chaincode deployed."

# 5. Verify deployment
echo "[5/5] Verifying chaincode..."
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE="${SAMPLES_DIR}/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${SAMPLES_DIR}/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
export CORE_PEER_ADDRESS=localhost:7051

# Query chaincode to verify it's operational
peer chaincode query \
    -C "$CHANNEL_NAME" \
    -n "$CC_NAME" \
    -c '{"function":"VaultStatus","Args":["test_uid"]}' 2>&1 || true

echo ""
echo "============================================"
echo " Fabric test-network READY"
echo " Channel: $CHANNEL_NAME"
echo " Chaincode: $CC_NAME"
echo " Peer0 Org1: localhost:7051"
echo " Peer0 Org2: localhost:9051"
echo " Orderer:    localhost:7050"
echo "============================================"

# Save connection info
cat > "${FABRIC_DIR}/connection_info.json" << EOF
{
    "channel": "$CHANNEL_NAME",
    "chaincode": "$CC_NAME",
    "peer_org1": "localhost:7051",
    "peer_org2": "localhost:9051",
    "orderer": "localhost:7050",
    "fabric_samples_dir": "$SAMPLES_DIR",
    "tls_cert_org1": "${SAMPLES_DIR}/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt",
    "msp_org1": "${SAMPLES_DIR}/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
}
EOF
echo "Connection info saved to blockchain/fabric/connection_info.json"
