// SPDX-License-Identifier: MIT
// VaultRegistry — D-IBFV on-chain CID management
// Stores IPFS CIDs for Shamir vault shares with threshold metadata.
//
// Designed for deployment on:
//   - Ethereum L1 (Sepolia testnet, ~12s blocks)
//   - Polygon PoS  (Amoy testnet, ~2s blocks)
//   - Local Hardhat/Ganache for development

pragma solidity ^0.8.20;

contract VaultRegistry {
    struct VaultRecord {
        string[] cids;       // IPFS CIDs of Shamir shares
        uint8 threshold;     // Shamir reconstruction threshold t
        uint8 shareCount;    // Total shares n
        bool revoked;        // Revocation flag (immutable once set)
        uint256 timestamp;   // Block timestamp at enrollment
        address owner;       // Address that enrolled the vault
    }

    // uid => VaultRecord
    mapping(bytes32 => VaultRecord) private vaults;

    // Existence check
    mapping(bytes32 => bool) private exists;

    event VaultStored(bytes32 indexed uid, uint8 threshold, uint8 shareCount, uint256 timestamp);
    event VaultRevoked(bytes32 indexed uid, uint256 timestamp);

    modifier onlyOwner(bytes32 uid) {
        require(exists[uid], "Vault does not exist");
        require(vaults[uid].owner == msg.sender, "Not vault owner");
        _;
    }

    /// @notice Store IPFS CIDs for a new vault enrollment.
    /// @param uid   Unique user/vault identifier (e.g., keccak256 of user ID).
    /// @param cids  Array of IPFS CID strings (one per Shamir share).
    /// @param t     Shamir threshold (minimum shares for reconstruction).
    /// @param n     Total number of shares.
    function storeVaultCIDs(
        bytes32 uid,
        string[] calldata cids,
        uint8 t,
        uint8 n
    ) external {
        require(!exists[uid] || vaults[uid].revoked, "Vault already exists and is active");
        require(cids.length == n, "CID count must equal n");
        require(t >= 2 && t <= n, "Invalid threshold");
        require(n <= 20, "Too many shares");

        // If re-enrolling after revocation, clear old record
        if (exists[uid]) {
            delete vaults[uid];
        }

        VaultRecord storage v = vaults[uid];
        for (uint256 i = 0; i < cids.length; i++) {
            v.cids.push(cids[i]);
        }
        v.threshold = t;
        v.shareCount = n;
        v.revoked = false;
        v.timestamp = block.timestamp;
        v.owner = msg.sender;
        exists[uid] = true;

        emit VaultStored(uid, t, n, block.timestamp);
    }

    /// @notice Look up the CIDs and Shamir parameters for a vault.
    /// @param uid  The vault identifier.
    /// @return cids Array of IPFS CID strings.
    /// @return t    Shamir threshold.
    /// @return n    Total share count.
    function lookupVaultCIDs(bytes32 uid)
        external
        view
        returns (string[] memory cids, uint8 t, uint8 n)
    {
        require(exists[uid], "Vault does not exist");
        VaultRecord storage v = vaults[uid];
        require(!v.revoked, "Vault has been revoked");
        return (v.cids, v.threshold, v.shareCount);
    }

    /// @notice Revoke a vault (only the original owner can revoke).
    /// @param uid  The vault identifier.
    function revokeVault(bytes32 uid) external onlyOwner(uid) {
        require(!vaults[uid].revoked, "Already revoked");
        vaults[uid].revoked = true;
        emit VaultRevoked(uid, block.timestamp);
    }

    /// @notice Check if a vault exists and its revocation status.
    /// @param uid  The vault identifier.
    /// @return isActive  True if vault exists and is not revoked.
    /// @return isRevoked True if vault exists and is revoked.
    function vaultStatus(bytes32 uid)
        external
        view
        returns (bool isActive, bool isRevoked)
    {
        if (!exists[uid]) return (false, false);
        return (!vaults[uid].revoked, vaults[uid].revoked);
    }
}
