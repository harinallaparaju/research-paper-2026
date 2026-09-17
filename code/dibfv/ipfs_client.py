"""
IPFS client for D-IBFV distributed vault share storage.

Provides upload/download/pin operations against the local IPFS HTTP API.
Each Shamir share is stored as a separate IPFS object identified by its CID.

Usage:
    client = IPFSClient()        # default http://127.0.0.1:5001
    cid = client.upload(share_bytes)
    data = client.download(cid)  # data == share_bytes
    client.pin(cid)              # ensure persistence
    client.unpin(cid)            # allow GC

All methods raise IPFSError on failure.
"""

import json
import time
import hashlib
from typing import List, Tuple, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError
from io import BytesIO


class IPFSError(Exception):
    """Raised on IPFS API failures."""
    pass


class IPFSClient:
    """Client for the IPFS HTTP API (Kubo)."""

    def __init__(self, api_url: str = "http://127.0.0.1:5001"):
        self.api_url = api_url.rstrip('/')

    def _post(self, endpoint: str, data: Optional[bytes] = None,
              content_type: Optional[str] = None) -> dict:
        """Send a POST request to the IPFS API."""
        url = f"{self.api_url}/api/v0/{endpoint}"
        req = Request(url, method='POST')
        if data is not None and content_type:
            req.add_header('Content-Type', content_type)
            req.data = data
        elif data is not None:
            req.data = data
        try:
            with urlopen(req, timeout=30) as resp:
                body = resp.read()
                return json.loads(body) if body else {}
        except URLError as e:
            raise IPFSError(f"IPFS API error on {endpoint}: {e}")
        except json.JSONDecodeError:
            return {}

    def _post_raw(self, endpoint: str) -> bytes:
        """POST and return raw bytes."""
        url = f"{self.api_url}/api/v0/{endpoint}"
        req = Request(url, method='POST')
        try:
            with urlopen(req, timeout=30) as resp:
                return resp.read()
        except URLError as e:
            raise IPFSError(f"IPFS API error on {endpoint}: {e}")

    def is_online(self) -> bool:
        """Check if the IPFS daemon is reachable."""
        try:
            result = self._post("version")
            return "Version" in result
        except IPFSError:
            return False

    def upload(self, data: bytes) -> str:
        """
        Upload bytes to IPFS.

        Args:
            data: Raw bytes to store.

        Returns:
            CID (Content Identifier) string, e.g. "Qm..."
        """
        # IPFS add API expects multipart/form-data
        boundary = b"----IPFSBoundary7MA4YWxkTrZu0gW"
        body = b"--" + boundary + b"\r\n"
        body += b'Content-Disposition: form-data; name="file"; filename="share"\r\n'
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += data + b"\r\n"
        body += b"--" + boundary + b"--\r\n"

        ct = f"multipart/form-data; boundary={boundary.decode()}"
        result = self._post("add?pin=true&quieter=true", body, ct)

        if "Hash" not in result:
            raise IPFSError(f"Upload failed: {result}")
        return result["Hash"]

    def download(self, cid: str) -> bytes:
        """
        Download bytes from IPFS by CID.

        Args:
            cid: Content identifier string.

        Returns:
            The raw bytes stored at that CID.
        """
        return self._post_raw(f"cat?arg={cid}")

    def pin(self, cid: str) -> None:
        """Pin a CID to prevent garbage collection."""
        result = self._post(f"pin/add?arg={cid}")
        if "Pins" not in result:
            raise IPFSError(f"Pin failed: {result}")

    def unpin(self, cid: str) -> None:
        """Unpin a CID to allow garbage collection."""
        try:
            self._post(f"pin/rm?arg={cid}")
        except IPFSError:
            pass  # Already unpinned

    def upload_shares(self, shares: List[Tuple[int, bytes]]) -> List[Tuple[int, str]]:
        """
        Upload multiple Shamir shares to IPFS.

        Args:
            shares: List of (index, share_bytes) from shamir.split().

        Returns:
            List of (index, cid) tuples.
        """
        results = []
        for idx, share_data in shares:
            cid = self.upload(share_data)
            results.append((idx, cid))
        return results

    def download_shares(self, share_cids: List[Tuple[int, str]]) -> List[Tuple[int, bytes]]:
        """
        Download multiple Shamir shares from IPFS.

        Args:
            share_cids: List of (index, cid) tuples.

        Returns:
            List of (index, share_bytes) tuples ready for shamir.reconstruct().
        """
        results = []
        for idx, cid in share_cids:
            data = self.download(cid)
            results.append((idx, data))
        return results
