#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Block implementation for the blockchain.
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional
import uuid

from blockchain.core.transaction import Transaction
from blockchain.utils.crypto import create_merkle_root

class Block:
    """Represents a block in the blockchain."""
    
    def __init__(self, 
                transactions: List[Transaction], 
                previous_hash: str, 
                timestamp: Optional[int] = None,
                block_number: Optional[int] = None,
                nonce: int = 0):
        """
        Initialize a new block.
        
        Args:
            transactions: List of transactions included in the block
            previous_hash: Hash of the previous block in the chain
            timestamp: Block creation timestamp (defaults to current time)
            block_number: Block height in the blockchain
            nonce: Arbitrary number used in mining process
        """
        self.block_id = str(uuid.uuid4())
        self.transactions = transactions
        self.transaction_count = len(transactions)
        self.previous_hash = previous_hash
        self.timestamp = timestamp or int(time.time() * 1000)  # milliseconds since epoch
        self.block_number = block_number
        self.nonce = nonce
        self.merkle_root = create_merkle_root([tx.to_dict() for tx in transactions])
        self.hash = self._calculate_hash()
        
        # AI-related metadata
        self.validator_confidence = None
        self.anomaly_scores = {}
        self.energy_usage = None
        
    def _calculate_hash(self) -> str:
        """
        Calculate the hash of the block.
        
        Returns:
            str: SHA-256 hash of the block
        """
        block_data = {
            'block_id': self.block_id,
            'transaction_count': self.transaction_count,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'block_number': self.block_number,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root
        }
        
        # Create a SHA-256 hash of the block data
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def set_ai_metadata(self, 
                       validator_confidence: float, 
                       anomaly_scores: Dict[str, float],
                       energy_usage: float) -> None:
        """
        Set AI-related metadata for the block.
        
        Args:
            validator_confidence: Confidence score from the predictive validator
            anomaly_scores: Anomaly scores for transactions in the block
            energy_usage: Energy consumed to create this block (in kWh)
        """
        self.validator_confidence = validator_confidence
        self.anomaly_scores = anomaly_scores
        self.energy_usage = energy_usage
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the block to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the block
        """
        return {
            'block_id': self.block_id,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'transaction_count': self.transaction_count,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'block_number': self.block_number,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root,
            'hash': self.hash,
            'validator_confidence': self.validator_confidence,
            'anomaly_scores': self.anomaly_scores,
            'energy_usage': self.energy_usage
        }
    
    def from_dict(self, data: Dict[str, Any]) -> 'Block':
        """
        Create a block from a dictionary representation.
        
        Args:
            data: Dictionary representation of a block
            
        Returns:
            Block: The created block instance
        """
        self.block_id = data['block_id']
        self.transactions = [Transaction().from_dict(tx) for tx in data['transactions']]
        self.transaction_count = data['transaction_count']
        self.previous_hash = data['previous_hash']
        self.timestamp = data['timestamp']
        self.block_number = data['block_number']
        self.nonce = data['nonce']
        self.merkle_root = data['merkle_root']
        self.hash = data['hash']
        self.validator_confidence = data.get('validator_confidence')
        self.anomaly_scores = data.get('anomaly_scores', {})
        self.energy_usage = data.get('energy_usage')
        return self
    
    def validate(self) -> bool:
        """
        Validate the integrity of the block.
        
        Returns:
            bool: True if the block is valid, False otherwise
        """
        # Check if the stored hash matches the calculated hash
        return self.hash == self._calculate_hash() 