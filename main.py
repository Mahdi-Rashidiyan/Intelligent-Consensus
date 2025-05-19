#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the AI-Enhanced Blockchain System for BRICS DeFi.
"""

import argparse
import logging
import os
import sys
import yaml
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import blockchain components
from blockchain.network import BlockchainNetwork, BlockchainNode
from blockchain.core.transaction import Transaction
from blockchain.utils.crypto import generate_key_pair, sign_transaction

# Import AI modules
from ai_modules.predictive_validation.model import PredictiveValidator
from ai_modules.resource_allocation.optimizer import ResourceOptimizer
from ai_modules.anomaly_detection.federated_detector import FederatedAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('blockchain.log')
    ]
)

logger = logging.getLogger(__name__)

class BRICSBlockchainApp:
    """Main application for the BRICS AI-Enhanced Blockchain."""
    
    def __init__(self, config_path: str):
        """
        Initialize the blockchain application.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.network = None
        self.predictive_validator = None
        self.resource_optimizer = None
        self.anomaly_detector = None
        
        logger.info("BRICS Blockchain Application initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def setup_network(self) -> None:
        """Set up the blockchain network."""
        network_id = self.config.get('blockchain', {}).get('network_id', 'brics-defi-testnet')
        self.network = BlockchainNetwork(network_id, self.config_path)
        
        # Add nodes from configuration
        nodes_config = self.config.get('blockchain', {}).get('nodes', [])
        for node_config in nodes_config:
            node = BlockchainNode(
                node_id=node_config['id'],
                country=node_config['country'],
                role=node_config['role'],
                endpoint=node_config.get('endpoint'),
                capacity=node_config.get('capacity', 'medium')
            )
            self.network.add_node(node)
        
        logger.info(f"Blockchain network set up with {len(self.network.nodes)} nodes")
    
    def setup_ai_components(self) -> None:
        """Set up AI components."""
        # Set up predictive validator
        pv_config = self.config.get('ai', {}).get('predictive_validation', {})
        model_path = pv_config.get('model_path')
        batch_size = pv_config.get('batch_size', 64)
        
        self.predictive_validator = PredictiveValidator(model_path, batch_size)
        self.network.set_predictive_validator(self.predictive_validator)
        
        # Set up resource optimizer
        ro_config = self.config.get('ai', {}).get('resource_allocation', {})
        learning_rate = ro_config.get('learning_rate', 0.0003)
        discount_factor = ro_config.get('discount_factor', 0.99)
        update_interval = ro_config.get('update_interval', 10000)
        
        self.resource_optimizer = ResourceOptimizer(learning_rate, discount_factor, update_interval)
        self.network.set_resource_optimizer(self.resource_optimizer)
        
        # Set up anomaly detector
        ad_config = self.config.get('ai', {}).get('anomaly_detection', {})
        input_dim = 20  # Default feature dimension
        num_clients = len(self.network.nodes)
        aggregation_strategy = ad_config.get('aggregation_strategy', 'fedavg')
        mu = ad_config.get('mu', 0.01)
        
        self.anomaly_detector = FederatedAnomalyDetector(input_dim, num_clients, aggregation_strategy, mu)
        self.network.set_anomaly_detector(self.anomaly_detector)
        
        logger.info("AI components set up")
    
    def start(self) -> None:
        """Start the blockchain network."""
        if self.network:
            self.network.start()
            logger.info("Blockchain network started")
        else:
            logger.error("Network not set up, call setup_network() first")
    
    def stop(self) -> None:
        """Stop the blockchain network."""
        if self.network:
            self.network.stop()
            logger.info("Blockchain network stopped")
        else:
            logger.error("Network not set up")
    
    def generate_test_transactions(self, count: int = 100) -> List[Transaction]:
        """
        Generate test transactions for the network.
        
        Args:
            count: Number of transactions to generate
            
        Returns:
            List[Transaction]: Generated transactions
        """
        transactions = []
        
        # Generate key pairs for test accounts
        accounts = {}
        for i in range(10):
            public_key, private_key = generate_key_pair()
            accounts[f"account-{i}"] = {
                "public_key": public_key,
                "private_key": private_key,
                "balance": 1000.0
            }
        
        # Generate random transactions
        for i in range(count):
            # Select random sender and receiver
            sender_idx = i % len(accounts)
            receiver_idx = (i + 1) % len(accounts)
            
            sender_id = f"account-{sender_idx}"
            receiver_id = f"account-{receiver_idx}"
            
            # Create transaction
            tx = Transaction(
                sender=accounts[sender_id]["public_key"],
                receiver=accounts[receiver_id]["public_key"],
                amount=10.0,
                currency="BRICS",
                data={
                    "fee": 0.1,
                    "gas_limit": 21000,
                    "nonce": i
                },
                tx_type="transfer"
            )
            
            # Sign transaction
            signature = sign_transaction(tx, accounts[sender_id]["private_key"])
            tx.sign(signature)
            
            transactions.append(tx)
        
        logger.info(f"Generated {len(transactions)} test transactions")
        return transactions
    
    def run_test_simulation(self, duration: int = 60) -> None:
        """
        Run a test simulation for a specified duration.
        
        Args:
            duration: Duration of the simulation in seconds
        """
        if not self.network:
            logger.error("Network not set up, call setup_network() first")
            return
        
        # Start the network
        self.start()
        
        # Generate test transactions
        transactions = self.generate_test_transactions(1000)
        
        # Add transactions to the network
        batch_size = 10
        batches = [transactions[i:i+batch_size] for i in range(0, len(transactions), batch_size)]
        
        start_time = time.time()
        end_time = start_time + duration
        
        batch_idx = 0
        while time.time() < end_time and batch_idx < len(batches):
            # Add a batch of transactions
            for tx in batches[batch_idx]:
                self.network.add_transaction(tx)
            
            batch_idx = (batch_idx + 1) % len(batches)
            
            # Log network status
            status = self.network.get_network_status()
            logger.info(f"Network status: {len(self.network.blockchain)} blocks, {len(self.network.transaction_pool)} pending transactions")
            
            # Sleep for a short time
            time.sleep(2)
        
        # Stop the network
        self.stop()
        
        # Log final statistics
        metrics = self.network.get_metrics()
        logger.info(f"Simulation completed. Final metrics: {metrics}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='BRICS AI-Enhanced Blockchain System')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['run', 'test', 'api'], default='test', help='Operation mode')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create and set up the application
    app = BRICSBlockchainApp(args.config)
    app.setup_network()
    app.setup_ai_components()
    
    # Run in the specified mode
    if args.mode == 'run':
        app.start()
        try:
            logger.info("Press Ctrl+C to stop the network")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            app.stop()
    elif args.mode == 'test':
        app.run_test_simulation(args.duration)
    elif args.mode == 'api':
        # This would start an API server in a real implementation
        logger.info("API mode not implemented yet")
    
    logger.info("Application terminated")

if __name__ == "__main__":
    main() 