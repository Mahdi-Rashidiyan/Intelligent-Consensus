#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Federated Anomaly Detection module for blockchain transactions.

This module implements a federated learning approach to detect anomalous transactions
across multiple nodes without centralizing sensitive transaction data.
"""

import logging
import numpy as np
import time
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import copy

logger = logging.getLogger(__name__)

class TransactionDataset(Dataset):
    """Dataset for transaction features."""
    
    def __init__(self, features: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features: Transaction features
        """
        self.features = torch.tensor(features, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Get the number of samples."""
        return len(self.features)
    
    def __getitem__(self, idx) -> torch.Tensor:
        """Get a sample."""
        return self.features[idx]

class AutoencoderModel(nn.Module):
    """Autoencoder model for anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Dimensionality of input features
            hidden_dims: List of hidden layer dimensions
        """
        super(AutoencoderModel, self).__init__()
        
        # Build encoder
        encoder_layers = []
        layer_dims = [input_dim] + hidden_dims
        
        for i in range(len(layer_dims) - 1):
            encoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.BatchNorm1d(layer_dims[i+1]))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        layer_dims.reverse()
        
        for i in range(len(layer_dims) - 1):
            decoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.BatchNorm1d(layer_dims[i+1]))
            else:
                decoder_layers.append(nn.Sigmoid())  # Output activation
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Reconstructed input
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Encoded representation
        """
        return self.encoder(x)

class FederatedClient:
    """Client for federated learning."""
    
    def __init__(self, client_id: str, input_dim: int):
        """
        Initialize a federated client.
        
        Args:
            client_id: Unique identifier for the client
            input_dim: Dimensionality of input features
        """
        self.client_id = client_id
        self.model = AutoencoderModel(input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.local_data = None
        self.training_history = []
        
        logger.info(f"Federated client {client_id} initialized")
    
    def set_data(self, features: np.ndarray) -> None:
        """
        Set the client's local data.
        
        Args:
            features: Transaction features
        """
        self.local_data = TransactionDataset(features)
    
    def train(self, epochs: int, batch_size: int = 32) -> Dict[str, float]:
        """
        Train the model on local data.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dict[str, float]: Training metrics
        """
        if self.local_data is None or len(self.local_data) == 0:
            logger.warning(f"Client {self.client_id} has no data for training")
            return {'loss': 0.0}
        
        dataloader = DataLoader(self.local_data, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            batch_losses = []
            
            for batch in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss = F.mse_loss(outputs, batch)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                batch_losses.append(loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            
            logger.debug(f"Client {self.client_id}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.training_history.append({
            'timestamp': time.time(),
            'epochs': epochs,
            'avg_loss': avg_loss
        })
        
        logger.info(f"Client {self.client_id} completed training with average loss: {avg_loss:.6f}")
        
        return {'loss': avg_loss}
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, torch.Tensor]: Model parameters
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Set the model parameters.
        
        Args:
            parameters: Model parameters
        """
        self.model.load_state_dict(parameters)
    
    def detect_anomalies(self, features: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Detect anomalies in the given features.
        
        Args:
            features: Transaction features
            threshold: Threshold for anomaly detection
            
        Returns:
            np.ndarray: Anomaly scores
        """
        self.model.eval()
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            reconstructions = self.model(features_tensor)
            
            # Calculate reconstruction error
            mse = torch.mean((reconstructions - features_tensor) ** 2, dim=1)
            
            # Convert to numpy array
            anomaly_scores = mse.cpu().numpy()
        
        return anomaly_scores

class FederatedAnomalyDetector:
    """
    Federated anomaly detection using collaborative learning.
    """
    
    def __init__(self, 
                input_dim: int = 20, 
                num_clients: int = 5,
                aggregation_strategy: str = "fedavg",
                mu: float = 0.01):
        """
        Initialize the federated anomaly detector.
        
        Args:
            input_dim: Dimensionality of input features
            num_clients: Number of federated clients
            aggregation_strategy: Strategy for aggregating models ("fedavg" or "fedprox")
            mu: Proximal term coefficient for FedProx
        """
        self.input_dim = input_dim
        self.num_clients = num_clients
        self.aggregation_strategy = aggregation_strategy
        self.mu = mu
        
        # Initialize global model
        self.global_model = AutoencoderModel(input_dim)
        
        # Initialize clients
        self.clients = {}
        for i in range(num_clients):
            client_id = f"client-{i+1}"
            self.clients[client_id] = FederatedClient(client_id, input_dim)
        
        # Training history
        self.training_rounds = 0
        self.global_history = []
        
        logger.info(f"Federated anomaly detector initialized with {num_clients} clients")
    
    def distribute_data(self, features_by_client: Dict[str, np.ndarray]) -> None:
        """
        Distribute data to clients.
        
        Args:
            features_by_client: Dictionary mapping client IDs to feature arrays
        """
        for client_id, features in features_by_client.items():
            if client_id in self.clients:
                self.clients[client_id].set_data(features)
                logger.debug(f"Distributed {len(features)} samples to client {client_id}")
            else:
                logger.warning(f"Client {client_id} not found")
    
    def train_round(self, 
                   client_ids: Optional[List[str]] = None, 
                   local_epochs: int = 5,
                   batch_size: int = 32) -> Dict[str, Any]:
        """
        Conduct a round of federated training.
        
        Args:
            client_ids: List of client IDs to participate in this round
            local_epochs: Number of local epochs per client
            batch_size: Batch size for training
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        # Select clients for this round
        if client_ids is None:
            client_ids = list(self.clients.keys())
            if len(client_ids) > self.num_clients:
                client_ids = np.random.choice(client_ids, self.num_clients, replace=False).tolist()
        
        # Distribute global model to selected clients
        global_params = self.global_model.state_dict()
        for client_id in client_ids:
            self.clients[client_id].set_model_parameters(copy.deepcopy(global_params))
        
        # Train on each client
        client_metrics = {}
        for client_id in client_ids:
            metrics = self.clients[client_id].train(local_epochs, batch_size)
            client_metrics[client_id] = metrics
        
        # Aggregate models
        if self.aggregation_strategy == "fedavg":
            self._aggregate_fedavg(client_ids)
        elif self.aggregation_strategy == "fedprox":
            self._aggregate_fedprox(client_ids)
        else:
            logger.warning(f"Unknown aggregation strategy: {self.aggregation_strategy}, using FedAvg")
            self._aggregate_fedavg(client_ids)
        
        # Update training history
        self.training_rounds += 1
        avg_loss = sum(metrics['loss'] for metrics in client_metrics.values()) / len(client_metrics)
        self.global_history.append({
            'round': self.training_rounds,
            'timestamp': time.time(),
            'num_clients': len(client_ids),
            'avg_loss': avg_loss
        })
        
        logger.info(f"Completed training round {self.training_rounds} with {len(client_ids)} clients, average loss: {avg_loss:.6f}")
        
        return {
            'round': self.training_rounds,
            'num_clients': len(client_ids),
            'client_metrics': client_metrics,
            'avg_loss': avg_loss
        }
    
    def _aggregate_fedavg(self, client_ids: List[str]) -> None:
        """
        Aggregate models using FedAvg algorithm.
        
        Args:
            client_ids: List of client IDs to aggregate
        """
        # Get client models
        client_params = [self.clients[client_id].get_model_parameters() for client_id in client_ids]
        
        # Average the parameters
        avg_params = {}
        for key in client_params[0].keys():
            avg_params[key] = torch.stack([params[key] for params in client_params]).mean(dim=0)
        
        # Update global model
        self.global_model.load_state_dict(avg_params)
        
        logger.debug(f"Aggregated models from {len(client_ids)} clients using FedAvg")
    
    def _aggregate_fedprox(self, client_ids: List[str]) -> None:
        """
        Aggregate models using FedProx algorithm.
        
        Args:
            client_ids: List of client IDs to aggregate
        """
        # Get client models and global model
        client_params = [self.clients[client_id].get_model_parameters() for client_id in client_ids]
        global_params = self.global_model.state_dict()
        
        # Average the parameters with proximal term
        avg_params = {}
        for key in client_params[0].keys():
            # Calculate average of client parameters
            client_avg = torch.stack([params[key] for params in client_params]).mean(dim=0)
            
            # Apply proximal term (weighted average with global model)
            avg_params[key] = (1 - self.mu) * client_avg + self.mu * global_params[key]
        
        # Update global model
        self.global_model.load_state_dict(avg_params)
        
        logger.debug(f"Aggregated models from {len(client_ids)} clients using FedProx with mu={self.mu}")
    
    def detect_anomalies(self, features: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Detect anomalies using the global model.
        
        Args:
            features: Transaction features
            threshold: Threshold for anomaly detection
            
        Returns:
            np.ndarray: Anomaly scores
        """
        self.global_model.eval()
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            reconstructions = self.global_model(features_tensor)
            
            # Calculate reconstruction error
            mse = torch.mean((reconstructions - features_tensor) ** 2, dim=1)
            
            # Convert to numpy array
            anomaly_scores = mse.cpu().numpy()
        
        return anomaly_scores
    
    def save_model(self, path: str) -> None:
        """
        Save the global model to a file.
        
        Args:
            path: Path to save the model
        """
        try:
            torch.save(self.global_model.state_dict(), path)
            logger.info(f"Saved global model to {path}")
        except Exception as e:
            logger.error(f"Error saving global model: {str(e)}")
    
    def load_model(self, path: str) -> bool:
        """
        Load the global model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.global_model.load_state_dict(torch.load(path))
            logger.info(f"Loaded global model from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading global model: {str(e)}")
            return False
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics for the detector.
        
        Returns:
            Dict[str, Any]: Training metrics
        """
        if not self.global_history:
            return {
                'rounds': 0,
                'avg_loss': 0.0,
                'loss_trend': []
            }
        
        return {
            'rounds': self.training_rounds,
            'avg_loss': self.global_history[-1]['avg_loss'],
            'loss_trend': [entry['avg_loss'] for entry in self.global_history]
        } 