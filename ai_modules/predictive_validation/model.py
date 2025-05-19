#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predictive Validation AI module for transaction validation.

This module uses deep learning to predict the likelihood of a transaction being valid
based on historical patterns, reducing computational requirements for full validation.
"""

import logging
import time
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class TransactionFeatureExtractor:
    """Extract features from transactions for AI validation."""
    
    def __init__(self, feature_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the transaction feature extractor.
        
        Args:
            feature_config: Configuration for feature extraction
        """
        self.feature_config = feature_config or {}
        self.sender_history = {}
        self.receiver_history = {}
        self.address_embeddings = {}
        
        # Feature normalization parameters
        self.amount_mean = 0.0
        self.amount_std = 1.0
        self.fee_mean = 0.0
        self.fee_std = 1.0
        
        logger.info("Transaction feature extractor initialized")
    
    def extract_features(self, transaction) -> np.ndarray:
        """
        Extract features from a transaction for AI processing.
        
        Args:
            transaction: Transaction to extract features from
            
        Returns:
            np.ndarray: Feature vector for the transaction
        """
        features = []
        
        # Transaction type (one-hot encoded)
        tx_type_features = [0] * 5  # Assuming 5 possible transaction types
        type_index = {
            "transfer": 0,
            "contract_call": 1,
            "token_swap": 2,
            "coinbase": 3,
            "governance": 4
        }.get(transaction.tx_type, 0)
        tx_type_features[type_index] = 1
        features.extend(tx_type_features)
        
        # Sender history features
        sender_features = self._get_sender_features(transaction.sender)
        features.extend(sender_features)
        
        # Receiver history features
        receiver_features = self._get_receiver_features(transaction.receiver)
        features.extend(receiver_features)
        
        # Amount features
        if transaction.amount is not None:
            normalized_amount = (transaction.amount - self.amount_mean) / (self.amount_std + 1e-10)
            features.append(normalized_amount)
            features.append(1.0 if transaction.amount > 0 else 0.0)
        else:
            features.extend([0.0, 0.0])  # Default values if amount is None
        
        # Fee features
        fee = transaction.data.get("fee", 0.0)
        normalized_fee = (fee - self.fee_mean) / (self.fee_std + 1e-10)
        features.append(normalized_fee)
        
        # Gas limit features
        gas_limit = transaction.data.get("gas_limit", 0)
        features.append(min(gas_limit / 1000000, 10.0))  # Normalize gas limit
        
        # Data size features
        data_size = len(json.dumps(transaction.data))
        features.append(min(data_size / 10000, 5.0))  # Normalize data size
        
        # Time features
        hour_of_day = (transaction.timestamp // 3600000) % 24
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        features.extend([hour_sin, hour_cos])
        
        # Convert to numpy array
        return np.array(features, dtype=np.float32)
    
    def update_history(self, transaction) -> None:
        """
        Update sender and receiver history based on a transaction.
        
        Args:
            transaction: Transaction to update history with
        """
        sender = transaction.sender
        receiver = transaction.receiver
        
        # Update sender history
        if sender not in self.sender_history:
            self.sender_history[sender] = {
                "count": 0,
                "total_amount": 0.0,
                "last_transaction_time": 0,
                "transactions": []
            }
        
        self.sender_history[sender]["count"] += 1
        self.sender_history[sender]["total_amount"] += transaction.amount or 0.0
        self.sender_history[sender]["last_transaction_time"] = transaction.timestamp
        self.sender_history[sender]["transactions"].append(transaction.tx_id)
        
        # Limit transaction history length
        if len(self.sender_history[sender]["transactions"]) > 100:
            self.sender_history[sender]["transactions"] = self.sender_history[sender]["transactions"][-100:]
        
        # Update receiver history
        if receiver not in self.receiver_history:
            self.receiver_history[receiver] = {
                "count": 0,
                "total_amount": 0.0,
                "last_transaction_time": 0,
                "transactions": []
            }
        
        self.receiver_history[receiver]["count"] += 1
        self.receiver_history[receiver]["total_amount"] += transaction.amount or 0.0
        self.receiver_history[receiver]["last_transaction_time"] = transaction.timestamp
        self.receiver_history[receiver]["transactions"].append(transaction.tx_id)
        
        # Limit transaction history length
        if len(self.receiver_history[receiver]["transactions"]) > 100:
            self.receiver_history[receiver]["transactions"] = self.receiver_history[receiver]["transactions"][-100:]
    
    def _get_sender_features(self, sender: str) -> List[float]:
        """Get features related to the sender address."""
        if sender not in self.sender_history:
            return [0.0] * 5  # Default values for new senders
        
        history = self.sender_history[sender]
        
        # Basic sender features
        count = min(history["count"] / 1000, 1.0)  # Normalize count
        total_amount = min(history["total_amount"] / 1000000, 1.0)  # Normalize total amount
        time_since_last = min((time.time() * 1000 - history["last_transaction_time"]) / (24 * 3600 * 1000), 30.0)  # Days
        
        # Sequence-based features would be more complex in a real implementation
        # Here we just use simple features
        
        return [count, total_amount, time_since_last, 0.0, 0.0]
    
    def _get_receiver_features(self, receiver: str) -> List[float]:
        """Get features related to the receiver address."""
        if receiver not in self.receiver_history:
            return [0.0] * 5  # Default values for new receivers
        
        history = self.receiver_history[receiver]
        
        # Basic receiver features
        count = min(history["count"] / 1000, 1.0)  # Normalize count
        total_amount = min(history["total_amount"] / 1000000, 1.0)  # Normalize total amount
        time_since_last = min((time.time() * 1000 - history["last_transaction_time"]) / (24 * 3600 * 1000), 30.0)  # Days
        
        # Sequence-based features would be more complex in a real implementation
        # Here we just use simple features
        
        return [count, total_amount, time_since_last, 0.0, 0.0]

class TransactionValidatorNetwork(nn.Module):
    """Neural network for transaction validation prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        """
        Initialize the transaction validator network.
        
        Args:
            input_dim: Dimensionality of input features
            hidden_dims: List of hidden layer dimensions
        """
        super(TransactionValidatorNetwork, self).__init__()
        
        # Build the layers
        layers = []
        layer_dims = [input_dim] + hidden_dims
        
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(layer_dims[i+1]))
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(layer_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

class TransactionAnomalyDetector(nn.Module):
    """Neural network for transaction anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 16):
        """
        Initialize the autoencoder for anomaly detection.
        
        Args:
            input_dim: Dimensionality of input features
            encoding_dim: Dimensionality of the encoded representation
        """
        super(TransactionAnomalyDetector, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode the input."""
        return self.encoder(x)

class PredictiveValidator:
    """
    Predictive Validator using deep learning for transaction validation.
    """
    
    def __init__(self, model_path: Optional[str] = None, batch_size: int = 64):
        """
        Initialize the predictive transaction validator.
        
        Args:
            model_path: Path to pre-trained model file
            batch_size: Batch size for inference
        """
        self.batch_size = batch_size
        self.feature_extractor = TransactionFeatureExtractor()
        
        # Default input dimension based on feature extractor
        input_dim = 22  # Should match the output dimension of feature_extractor
        
        # Create models
        self.validator_model = TransactionValidatorNetwork(input_dim)
        self.anomaly_detector = TransactionAnomalyDetector(input_dim)
        
        # Load pre-trained models if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.warning(f"Model file not found at {model_path}, using untrained models")
        
        # Set to evaluation mode
        self.validator_model.eval()
        self.anomaly_detector.eval()
        
        logger.info("Predictive validator initialized")
    
    def _load_model(self, model_path: str) -> None:
        """
        Load pre-trained models from file.
        
        Args:
            model_path: Path to pre-trained model file
        """
        try:
            checkpoint = torch.load(model_path)
            self.validator_model.load_state_dict(checkpoint['validator_model'])
            self.anomaly_detector.load_state_dict(checkpoint['anomaly_detector'])
            self.feature_extractor.amount_mean = checkpoint.get('amount_mean', 0.0)
            self.feature_extractor.amount_std = checkpoint.get('amount_std', 1.0)
            self.feature_extractor.fee_mean = checkpoint.get('fee_mean', 0.0)
            self.feature_extractor.fee_std = checkpoint.get('fee_std', 1.0)
            logger.info(f"Loaded pre-trained models from {model_path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def predict_validity(self, transaction) -> float:
        """
        Predict the validity of a transaction.
        
        Args:
            transaction: Transaction to validate
            
        Returns:
            float: Confidence score for transaction validity (0.0 to 1.0)
        """
        # Extract features for the transaction
        features = self.feature_extractor.extract_features(transaction)
        features_tensor = torch.tensor(features).float().unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            confidence = self.validator_model(features_tensor).item()
        
        # Update transaction history
        self.feature_extractor.update_history(transaction)
        
        return confidence
    
    def check_anomalies(self, transaction) -> float:
        """
        Check if a transaction is anomalous.
        
        Args:
            transaction: Transaction to check for anomalies
            
        Returns:
            float: Anomaly score (higher means more anomalous)
        """
        # Extract features for the transaction
        features = self.feature_extractor.extract_features(transaction)
        features_tensor = torch.tensor(features).float().unsqueeze(0)  # Add batch dimension
        
        # Use autoencoder for anomaly detection
        with torch.no_grad():
            reconstruction = self.anomaly_detector(features_tensor)
            
            # Calculate reconstruction error (MSE)
            mse = F.mse_loss(reconstruction, features_tensor).item()
            
            # Scale the error to get an anomaly score between 0 and 1
            anomaly_score = min(mse * 10, 1.0)  # Arbitrary scaling for demo
        
        return anomaly_score
    
    def batch_predict(self, transactions) -> Tuple[List[float], List[float]]:
        """
        Predict validity and anomaly scores for a batch of transactions.
        
        Args:
            transactions: List of transactions to process
            
        Returns:
            Tuple[List[float], List[float]]: Validity confidence scores and anomaly scores
        """
        features_list = [self.feature_extractor.extract_features(tx) for tx in transactions]
        features_tensor = torch.tensor(np.stack(features_list)).float()
        
        validity_scores = []
        anomaly_scores = []
        
        # Process in batches
        for i in range(0, len(features_list), self.batch_size):
            batch = features_tensor[i:i+self.batch_size]
            
            with torch.no_grad():
                # Predict validity
                batch_validity = self.validator_model(batch).squeeze().tolist()
                if isinstance(batch_validity, float):  # Handle single-element batch
                    validity_scores.append(batch_validity)
                else:
                    validity_scores.extend(batch_validity)
                
                # Predict anomalies
                batch_recon = self.anomaly_detector(batch)
                # Calculate reconstruction error for each transaction
                for j in range(len(batch)):
                    mse = F.mse_loss(batch_recon[j], batch[j]).item()
                    anomaly_scores.append(min(mse * 10, 1.0))  # Scale to 0-1
        
        # Update transaction history
        for tx in transactions:
            self.feature_extractor.update_history(tx)
        
        return validity_scores, anomaly_scores
    
    def save_model(self, model_path: str) -> None:
        """
        Save the models to file.
        
        Args:
            model_path: Path to save the model file
        """
        try:
            checkpoint = {
                'validator_model': self.validator_model.state_dict(),
                'anomaly_detector': self.anomaly_detector.state_dict(),
                'amount_mean': self.feature_extractor.amount_mean,
                'amount_std': self.feature_extractor.amount_std,
                'fee_mean': self.feature_extractor.fee_mean,
                'fee_std': self.feature_extractor.fee_std
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Saved models to {model_path}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def evaluate_performance(self, transactions, true_labels) -> Dict[str, float]:
        """
        Evaluate the performance of the predictive validator.
        
        Args:
            transactions: List of transactions to evaluate
            true_labels: List of true labels (1 for valid, 0 for invalid)
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        validity_scores, _ = self.batch_predict(transactions)
        
        # Calculate metrics
        predictions = [1 if score >= 0.5 else 0 for score in validity_scores]
        
        # Accuracy
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(true_labels) if len(true_labels) > 0 else 0.0
        
        # Precision, recall, F1 (for valid transactions)
        true_positives = sum(1 for pred, true in zip(predictions, true_labels) if pred == 1 and true == 1)
        false_positives = sum(1 for pred, true in zip(predictions, true_labels) if pred == 1 and true == 0)
        false_negatives = sum(1 for pred, true in zip(predictions, true_labels) if pred == 0 and true == 1)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        } 