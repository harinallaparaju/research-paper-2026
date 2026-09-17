// Package main implements the D-IBFV Vault Registry chaincode for
// Hyperledger Fabric. It stores IPFS CIDs of Shamir vault shares
// on the ledger with threshold metadata and revocation support.
//
// Chaincode functions:
//   StoreVaultCIDs  — enroll a new vault (write)
//   LookupVaultCIDs — retrieve CIDs + params (read)
//   RevokeVault     — mark vault as revoked (write)
//   VaultStatus     — check existence/revocation (read)

package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// VaultRecord stored on the ledger.
type VaultRecord struct {
	CIDs       []string `json:"cids"`
	Threshold  int      `json:"threshold"`
	ShareCount int      `json:"shareCount"`
	Revoked    bool     `json:"revoked"`
	Timestamp  string   `json:"timestamp"`
	Owner      string   `json:"owner"`
}

// VaultRegistryContract implements the chaincode.
type VaultRegistryContract struct {
	contractapi.Contract
}

// StoreVaultCIDs enrolls a new vault on the ledger.
func (c *VaultRegistryContract) StoreVaultCIDs(
	ctx contractapi.TransactionContextInterface,
	uid string,
	cidsJSON string,
	t int,
	n int,
) error {
	// Validate parameters
	if t < 2 || t > n {
		return fmt.Errorf("invalid threshold: t=%d, n=%d", t, n)
	}
	if n > 20 {
		return fmt.Errorf("too many shares: n=%d (max 20)", n)
	}

	var cids []string
	if err := json.Unmarshal([]byte(cidsJSON), &cids); err != nil {
		return fmt.Errorf("invalid CIDs JSON: %v", err)
	}
	if len(cids) != n {
		return fmt.Errorf("CID count %d != n=%d", len(cids), n)
	}

	// Check if vault already exists and is active
	existing, err := ctx.GetStub().GetState(uid)
	if err != nil {
		return fmt.Errorf("failed to read state: %v", err)
	}
	if existing != nil {
		var old VaultRecord
		if err := json.Unmarshal(existing, &old); err == nil {
			if !old.Revoked {
				return fmt.Errorf("vault %s already exists and is active", uid)
			}
		}
	}

	// Get transaction creator identity
	creator, err := ctx.GetClientIdentity().GetID()
	if err != nil {
		creator = "unknown"
	}

	txTime, err := ctx.GetStub().GetTxTimestamp()
	ts := time.Now().UTC().Format(time.RFC3339)
	if err == nil && txTime != nil {
		ts = time.Unix(txTime.Seconds, int64(txTime.Nanos)).UTC().Format(time.RFC3339)
	}

	record := VaultRecord{
		CIDs:       cids,
		Threshold:  t,
		ShareCount: n,
		Revoked:    false,
		Timestamp:  ts,
		Owner:      creator,
	}

	recordJSON, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %v", err)
	}

	return ctx.GetStub().PutState(uid, recordJSON)
}

// LookupVaultCIDs retrieves the CIDs and Shamir parameters for a vault.
func (c *VaultRegistryContract) LookupVaultCIDs(
	ctx contractapi.TransactionContextInterface,
	uid string,
) (*VaultRecord, error) {
	data, err := ctx.GetStub().GetState(uid)
	if err != nil {
		return nil, fmt.Errorf("failed to read state: %v", err)
	}
	if data == nil {
		return nil, fmt.Errorf("vault %s does not exist", uid)
	}

	var record VaultRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, fmt.Errorf("failed to unmarshal record: %v", err)
	}
	if record.Revoked {
		return nil, fmt.Errorf("vault %s has been revoked", uid)
	}

	return &record, nil
}

// RevokeVault marks a vault as revoked.
func (c *VaultRegistryContract) RevokeVault(
	ctx contractapi.TransactionContextInterface,
	uid string,
) error {
	data, err := ctx.GetStub().GetState(uid)
	if err != nil {
		return fmt.Errorf("failed to read state: %v", err)
	}
	if data == nil {
		return fmt.Errorf("vault %s does not exist", uid)
	}

	var record VaultRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return fmt.Errorf("failed to unmarshal record: %v", err)
	}
	if record.Revoked {
		return fmt.Errorf("vault %s already revoked", uid)
	}

	// Verify caller is the owner
	caller, err := ctx.GetClientIdentity().GetID()
	if err == nil && caller != record.Owner {
		return fmt.Errorf("only the vault owner can revoke")
	}

	record.Revoked = true
	recordJSON, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %v", err)
	}

	return ctx.GetStub().PutState(uid, recordJSON)
}

// VaultStatus checks existence and revocation.
func (c *VaultRegistryContract) VaultStatus(
	ctx contractapi.TransactionContextInterface,
	uid string,
) (string, error) {
	data, err := ctx.GetStub().GetState(uid)
	if err != nil {
		return "", fmt.Errorf("failed to read state: %v", err)
	}
	if data == nil {
		return `{"exists":false,"revoked":false}`, nil
	}

	var record VaultRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return "", fmt.Errorf("failed to unmarshal: %v", err)
	}

	result := fmt.Sprintf(`{"exists":true,"revoked":%v}`, record.Revoked)
	return result, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&VaultRegistryContract{})
	if err != nil {
		panic(fmt.Sprintf("Error creating chaincode: %v", err))
	}
	if err := chaincode.Start(); err != nil {
		panic(fmt.Sprintf("Error starting chaincode: %v", err))
	}
}
