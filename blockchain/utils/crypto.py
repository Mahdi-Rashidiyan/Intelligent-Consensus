#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cryptographic utilities for the blockchain.
"""

import hashlib
import json
import secrets
from typing import Dict, List, Any, Tuple

def generate_key_pair() -> Tuple[str, str]:
    """
    Generate a key pair for signing transactions.
    
    In a real implementation, this would use proper cryptographic libraries.
    Here we just simulate the process with random strings.
    
    Returns:
        Tuple[str, str]: Public key and private key
    """
    # Generate a random private key
    private_key = secrets.token_hex(32)
    
    # Derive public key from private key
    public_key = hashlib.sha256(private_key.encode()).hexdigest()
    
    return public_key, private_key

def sign_transaction(transaction, private_key: str) -> str:
    """
    Sign a transaction with a private key.
    
    In a real implementation, this would use proper cryptographic libraries.
    Here we just simulate the process.
    
    Args:
        transaction: The transaction to sign
        private_key: The private key to sign with
        
    Returns:
        str: The signature
    """
    # Create a string representation of the transaction
    tx_data = {
        'tx_id': transaction.tx_id,
        'sender': transaction.sender,
        'receiver': transaction.receiver,
        'amount': transaction.amount,
        'currency': transaction.currency,
        'data': transaction.data,
        'timestamp': transaction.timestamp,
        'tx_type': transaction.tx_type
    }
    
    tx_string = json.dumps(tx_data, sort_keys=True)
    
    # Create a signature by hashing the transaction data with the private key
    signature_input = tx_string + private_key
    signature = hashlib.sha256(signature_input.encode()).hexdigest()
    
    return signature

def verify_signature(transaction) -> bool:
    """
    Verify the signature of a transaction.
    
    In a real implementation, this would use proper cryptographic libraries.
    Here we just simulate the process.
    
    Args:
        transaction: The transaction to verify
        
    Returns:
        bool: True if the signature is valid, False otherwise
    """
    # For simulation purposes, we'll just assume the signature is valid
    # In a real implementation, this would verify the signature against the public key
    
    if not transaction.signature:
        return False
    
    # Special case for coinbase transactions
    if transaction.tx_type == "coinbase" and transaction.signature == "COINBASE_TRANSACTION":
        return True
    
    # For regular transactions, we'll just check if the signature exists
    # In a real implementation, we would verify the signature against the public key
    return len(transaction.signature) == 64  # Simple check for SHA-256 hash length

def create_merkle_root(items: List[Dict[str, Any]]) -> str:
    """
    Create a Merkle root from a list of items.
    
    Args:
        items: List of items to include in the Merkle tree
        
    Returns:
        str: Merkle root hash
    """
    if not items:
        return "0" * 64  # Empty Merkle root
    
    # Convert items to hashes
    hashes = [hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest() for item in items]
    
    # Build the Merkle tree
    while len(hashes) > 1:
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])  # Duplicate the last hash if odd number
        
        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i+1]
            next_hash = hashlib.sha256(combined.encode()).hexdigest()
            next_level.append(next_hash)
        
        hashes = next_level
    
    return hashes[0]  # Root hash 