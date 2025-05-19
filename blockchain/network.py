#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blockchain Network implementation with AI integration for BRICS DeFi testnet.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
import yaml
import os

from blockchain.core.block import Block
from blockchain.core.transaction import Transaction
from blockchain.consensus.ai_consensus import AIEnhancedConsensus
from blockchain.utils.crypto import verify_signature
from blockchain.utils.stats import NetworkStats

logger = logging.getLogger(__name__)

class BlockchainNode:
    """Represents a node in the blockchain network."""
    
    def __init__(self, node_id: str, country: str, role: str, endpoint: Optional[str] = None, 
                 capacity: str = "medium"):
        """
        Initialize a blockchain node.
        
        Args:
            node_id: Unique identifier for the node
            country: Country where the node is located (BRICS member)
            role: Role of the node (validator, observer)
            endpoint: Network endpoint for the node
            capacity: Computational capacity of the node (low, medium, high)
        """
        self.node_id = node_id
        self.country = country
        self.role = role
        self.endpoint = endpoint
        self.capacity = capacity
        self.status = "inactive"
        self.peers = []
        self.transaction_pool = []
        self.blockchain = []  # Local copy of the blockchain
        self.resource_usage = {
            "cpu": 0.0,
            "memory": 0.0,
            "network_in": 0.0,
            "network_out": 0.0,
            "disk": 0.0
        }
        
        # Initialize local state for AI components
        self.local_model_state = {}
        
        logger.info(f"Node {node_id} from {country} initialized with {capacity} capacity")
    
    def add_peer(self, peer_node_id: str) -> None:
        """Add a peer to this node's peer list."""
        if peer_node_id not in self.peers and peer_node_id != self.node_id:
            self.peers.append(peer_node_id)
            logger.debug(f"Node {self.node_id} added peer {peer_node_id}")
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to the node's transaction pool."""
        # Verify transaction signature and structure
        if verify_signature(transaction):
            self.transaction_pool.append(transaction)
            logger.debug(f"Node {self.node_id} added transaction {transaction.tx_id} to pool")
            return True
        else:
            logger.warning(f"Node {self.node_id} rejected invalid transaction")
            return False
    
    def update_resource_usage(self, metrics: Dict[str, float]) -> None:
        """Update the node's resource usage metrics."""
        self.resource_usage.update(metrics)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the node."""
        return {
            "node_id": self.node_id,
            "country": self.country,
            "role": self.role,
            "status": self.status,
            "peer_count": len(self.peers),
            "tx_pool_size": len(self.transaction_pool),
            "blockchain_height": len(self.blockchain),
            "resource_usage": self.resource_usage
        }


class BlockchainNetwork:
    """
    Represents the entire blockchain network with AI integration.
    """
    
    def __init__(self, network_id: str, config_path: Optional[str] = None):
        """
        Initialize the blockchain network.
        
        Args:
            network_id: Unique identifier for the network
            config_path: Path to the network configuration file
        """
        self.network_id = network_id
        self.nodes: Dict[str, BlockchainNode] = {}
        self.blockchain: List[Block] = []
        self.transaction_pool: List[Transaction] = []
        self.is_running = False
        self.stats = NetworkStats()
        
        # Default configuration
        self.config = {
            "block_time": 2000,  # milliseconds
            "max_tx_per_block": 1000,
            "consensus_protocol": "ai_enhanced_raft"
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config.update(yaml.safe_load(f).get('blockchain', {}))
        
        # AI components (will be set by the main application)
        self.predictive_validator = None
        self.resource_optimizer = None
        self.anomaly_detector = None
        
        # Initialize consensus mechanism
        self.consensus = AIEnhancedConsensus(self)
        
        logger.info(f"Blockchain network {network_id} initialized with {self.config['consensus_protocol']} consensus")
    
    def add_node(self, node: BlockchainNode) -> None:
        """Add a node to the network."""
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
            logger.info(f"Node {node.node_id} added to the network")
        else:
            logger.warning(f"Node {node.node_id} already exists in the network")
    
    def set_predictive_validator(self, validator) -> None:
        """Set the predictive transaction validator AI module."""
        self.predictive_validator = validator
        self.consensus.set_predictive_validator(validator)
        logger.info("Predictive transaction validator set for the network")
    
    def set_resource_optimizer(self, optimizer) -> None:
        """Set the resource optimization AI module."""
        self.resource_optimizer = optimizer
        logger.info("Resource optimizer set for the network")
    
    def set_anomaly_detector(self, detector) -> None:
        """Set the anomaly detection AI module."""
        self.anomaly_detector = detector
        logger.info("Anomaly detector set for the network")
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add a transaction to the network pool after validation.
        
        Args:
            transaction: The transaction to add
            
        Returns:
            bool: True if transaction was accepted, False otherwise
        """
        # First, use AI for pre-validation if available
        if self.predictive_validator:
            prediction = self.predictive_validator.predict_validity(transaction)
            if prediction < self.config.get('validation_threshold', 0.8):
                logger.warning(f"Transaction {transaction.tx_id} rejected by predictive validator (score: {prediction:.2f})")
                return False
        
        # Check for anomalies if detector is available
        if self.anomaly_detector:
            anomaly_score = self.anomaly_detector.detect_anomalies(transaction)
            if anomaly_score > self.config.get('anomaly_threshold', 0.7):
                logger.warning(f"Transaction {transaction.tx_id} flagged as potential anomaly (score: {anomaly_score:.2f})")
                # We still add it to the pool but flag it for further investigation
                transaction.set_flag("potential_anomaly", anomaly_score)
        
        # Add to transaction pool if it passes validation
        if verify_signature(transaction):
            self.transaction_pool.append(transaction)
            logger.debug(f"Transaction {transaction.tx_id} added to network pool")
            return True
        else:
            logger.warning(f"Transaction {transaction.tx_id} has invalid signature")
            return False
    
    def start(self) -> None:
        """Start the blockchain network."""
        if self.is_running:
            logger.warning("Network is already running")
            return
        
        self.is_running = True
        
        # Start the consensus process
        self.consensus_thread = threading.Thread(target=self._consensus_loop)
        self.consensus_thread.daemon = True
        self.consensus_thread.start()
        
        # Start resource optimization if available
        if self.resource_optimizer:
            self.optimization_thread = threading.Thread(target=self._optimization_loop)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
        
        # Activate all nodes
        for node_id, node in self.nodes.items():
            node.status = "active"
        
        logger.info(f"Blockchain network {self.network_id} started with {len(self.nodes)} nodes")
    
    def stop(self) -> None:
        """Stop the blockchain network."""
        self.is_running = False
        
        # Deactivate all nodes
        for node_id, node in self.nodes.items():
            node.status = "inactive"
        
        logger.info(f"Blockchain network {self.network_id} stopped")
    
    def _consensus_loop(self) -> None:
        """Main consensus loop."""
        while self.is_running:
            try:
                # Create a new block using the consensus mechanism
                new_block = self.consensus.create_block(self.transaction_pool[:self.config['max_tx_per_block']])
                
                if new_block:
                    # Add the block to the blockchain
                    self.blockchain.append(new_block)
                    
                    # Remove the transactions included in the block from the pool
                    tx_ids = [tx.tx_id for tx in new_block.transactions]
                    self.transaction_pool = [tx for tx in self.transaction_pool if tx.tx_id not in tx_ids]
                    
                    logger.info(f"New block added to the blockchain: {new_block.block_id}")
                    
                    # Update statistics
                    self.stats.update_block_metrics(new_block)
                
                # Sleep for the block time
                time.sleep(self.config['block_time'] / 1000)  # Convert to seconds
                
            except Exception as e:
                logger.error(f"Error in consensus loop: {str(e)}")
                time.sleep(5)  # Sleep and retry on error
    
    def _optimization_loop(self) -> None:
        """Resource optimization loop using AI."""
        while self.is_running:
            try:
                # Collect current resource usage from all nodes
                resource_states = {node_id: node.resource_usage for node_id, node in self.nodes.items()}
                
                # Get optimized resource allocation from the AI optimizer
                allocations = self.resource_optimizer.optimize(
                    resource_states, 
                    self.transaction_pool, 
                    self.stats.get_metrics()
                )
                
                # Apply the optimized allocations
                for node_id, allocation in allocations.items():
                    if node_id in self.nodes:
                        # In a real implementation, this would adjust computational resources
                        # For simulation, we just update the metrics
                        logger.debug(f"Optimizing resources for node {node_id}: {allocation}")
                
                # Sleep before next optimization cycle
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {str(e)}")
                time.sleep(5)  # Sleep and retry on error
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get the current status of the network."""
        active_nodes = sum(1 for node in self.nodes.values() if node.status == "active")
        
        return {
            "network_id": self.network_id,
            "nodes_total": len(self.nodes),
            "nodes_active": active_nodes,
            "blockchain_height": len(self.blockchain),
            "transaction_pool_size": len(self.transaction_pool),
            "consensus_protocol": self.config['consensus_protocol'],
            "is_running": self.is_running,
            "metrics": self.stats.get_metrics()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the network."""
        return self.stats.get_metrics() 