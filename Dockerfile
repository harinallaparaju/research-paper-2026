# ─────────────────────────────────────────────────────────────────────────────
# D-IBFV — Research Container
#
# What this image contains:
#   • Python 3.11 with all biometric + blockchain dependencies
#   • go-ipfs (Kubo) daemon — for vault share storage
#   • Ganache CLI — local Ethereum node
#   • Solidity compiler (via py-solc-x, auto-installed on first run)
#
# Build:   docker build -t dibfv .
# Run:     docker-compose up          (recommended — starts IPFS + Ganache too)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV runtime libraries
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        # Node.js (for Ganache CLI)
        nodejs \
        npm \
        # curl for IPFS download + healthchecks
        curl \
        wget \
        # Build tools (needed by some pip packages on Linux)
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Install Ganache CLI ───────────────────────────────────────────────────────
RUN npm install -g ganache --quiet

# ── Install IPFS (Kubo) ───────────────────────────────────────────────────────
RUN IPFS_VERSION=v0.27.0 && \
    wget -q "https://dist.ipfs.tech/kubo/${IPFS_VERSION}/kubo_${IPFS_VERSION}_linux-amd64.tar.gz" \
         -O /tmp/kubo.tar.gz && \
    tar -xzf /tmp/kubo.tar.gz -C /tmp && \
    mv /tmp/kubo/ipfs /usr/local/bin/ipfs && \
    rm -rf /tmp/kubo* && \
    ipfs --version

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download solc (Solidity compiler) so first run is instant
RUN python -c "import solcx; solcx.install_solc('0.8.20')" || true

# ── Copy project code ─────────────────────────────────────────────────────────
# NOTE: code/ is copied into the image so the scripts are available even if
#       the volume mount is not used. data/ is NOT copied — it is too large
#       (~1.3 GB) and is always provided via docker-compose volume mount.
COPY code/ ./code/

# Working directory for all experiment scripts
WORKDIR /app/code

# ── Default command: show help ─────────────────────────────────────────────────
CMD ["python", "-c", "print('D-IBFV container ready. See README for run commands.')"]
