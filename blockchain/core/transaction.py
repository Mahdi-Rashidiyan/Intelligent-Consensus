#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transaction implementation for the blockchain.
"""

import hashlib
import json
import time
import uuid
from typing import Dict, Any, List, Optional

class Transaction:
    """Represents a transaction in the blockchain."""
    
    def __init__(self, 
                sender: Optional[str] = None, 
                receiver: Optional[str] = None, 
                amount: Optional[float] = None, 
                currency: Optional[str] = None,
                data: Optional[Dict[str, Any]] = None,
                signature: Optional[str] = None,
                timestamp: Optional[int] = None,
                tx_type: Optional[str] = "transfer"):
        """
        Initialize a new transaction.
        
        Args:
            sender: Address of the sender
            receiver: Address of the receiver
            amount: Transaction amount
            currency: Currency of the transaction
            data: Additional transaction data
            signature: Digital signature of the transaction
            timestamp: Transaction creation timestamp
            tx_type: Type of transaction (transfer, contract_call, etc.)
        """
        self.tx_id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.currency = currency
        self.data = data or {}
        self.timestamp = timestamp or int(time.time() * 1000)  # milliseconds since epoch
        self.signature = signature
        self.tx_type = tx_type
        self.tx_hash = None
        
        # Flags for AI analysis
        self.flags = {}
        
        # Calculate transaction hash if we have required data
        if sender and receiver:
            self.tx_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """
        Calculate the hash of the transaction.
        
        Returns:
            str: SHA-256 hash of the transaction
        """
        tx_data = {
            'tx_id': self.tx_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'currency': self.currency,
            'data': self.data,
            'timestamp': self.timestamp,
            'tx_type': self.tx_type
        }
        
        # Create a SHA-256 hash of the transaction data
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def sign(self, signature: str) -> None:
        """
        Sign the transaction with the provided signature.
        
        Args:
            signature: Digital signature for the transaction
        """
        self.signature = signature
    
    def set_flag(self, flag_name: str, value: Any) -> None:
        """
        Set a flag on the transaction, typically used for AI analysis results.
        
        Args:
            flag_name: Name of the flag
            value: Value of the flag
        """
        self.flags[flag_name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the transaction to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the transaction
        """
        return {
            'tx_id': self.tx_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'currency': self.currency,
            'data': self.data,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'tx_type': self.tx_type,
            'tx_hash': self.tx_hash,
            'flags': self.flags
        }
    
    def from_dict(self, data: Dict[str, Any]) -> 'Transaction':
        """
        Create a transaction from a dictionary representation.
        
        Args:
            data: Dictionary representation of a transaction
            
        Returns:
            Transaction: The created transaction instance
        """
        self.tx_id = data['tx_id']
        self.sender = data['sender']
        self.receiver = data['receiver']
        self.amount = data['amount']
        self.currency = data['currency']
        self.data = data['data']
        self.timestamp = data['timestamp']
        self.signature = data['signature']
        self.tx_type = data['tx_type']
        self.tx_hash = data['tx_hash']
        self.flags = data.get('flags', {})
        return self
    
    def validate_structure(self) -> bool:
        """
        Validate the structure of the transaction.
        
        Returns:
            bool: True if the transaction structure is valid, False otherwise
        """
        # Basic structure validation
        if not self.sender or not self.receiver:
            return False
        
        # Amount validation for transfer transactions
        if self.tx_type == "transfer" and (self.amount is None or self.amount <= 0):
            return False
        
        # Hash validation
        if self.tx_hash != self._calculate_hash():
            return False
        
        return True
    
    @staticmethod
    def create_coinbase(receiver: str, amount: float, currency: str) -> 'Transaction':
        """
        Create a coinbase transaction (special transaction that creates new coins).
        
        Args:
            receiver: Address of the receiver
            amount: Transaction amount
            currency: Currency of the transaction
            
        Returns:
            Transaction: A new coinbase transaction
        """
        tx = Transaction(
            sender="0x0000000000000000000000000000000000000000",  # Conventional zero address
            receiver=receiver,
            amount=amount,
            currency=currency,
            tx_type="coinbase"
        )
        
        # Sign with a special "minted" signature
        tx.sign("COINBASE_TRANSACTION")
        
        return tx 