#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network statistics utilities for the blockchain.
"""

import time
from typing import Dict, Any, List
import statistics
from collections import deque

class NetworkStats:
    """Collects and calculates statistics for the blockchain network."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the network statistics collector.
        
        Args:
            max_history: Maximum number of blocks to keep in history
        """
        self.start_time = time.time()
        self.block_times = deque(maxlen=max_history)
        self.transaction_counts = deque(maxlen=max_history)
        self.block_sizes = deque(maxlen=max_history)
        self.energy_usage = deque(maxlen=max_history)
        self.validator_confidences = deque(maxlen=max_history)
        self.anomaly_scores = deque(maxlen=max_history)
        
        # Aggregate metrics
        self.total_blocks = 0
        self.total_transactions = 0
        self.total_energy = 0.0
        
        # Performance metrics
        self.last_block_time = self.start_time
    
    def update_block_metrics(self, block) -> None:
        """
        Update metrics with a new block.
        
        Args:
            block: The new block to update metrics with
        """
        # Calculate block time
        current_time = time.time()
        block_time = current_time - self.last_block_time
        self.block_times.append(block_time)
        self.last_block_time = current_time
        
        # Update transaction metrics
        tx_count = len(block.transactions)
        self.transaction_counts.append(tx_count)
        self.total_transactions += tx_count
        
        # Update block size metrics (approximated by transaction count)
        self.block_sizes.append(tx_count)
        
        # Update energy metrics
        energy = block.energy_usage or 0.0
        self.energy_usage.append(energy)
        self.total_energy += energy
        
        # Update AI metrics
        if block.validator_confidence is not None:
            self.validator_confidences.append(block.validator_confidence)
        
        # Update anomaly scores
        if block.anomaly_scores:
            avg_anomaly = sum(block.anomaly_scores.values()) / max(1, len(block.anomaly_scores))
            self.anomaly_scores.append(avg_anomaly)
        
        # Update total blocks
        self.total_blocks += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current network metrics.
        
        Returns:
            Dict[str, Any]: Current network metrics
        """
        # Calculate TPS (transactions per second)
        uptime = max(1, time.time() - self.start_time)
        tps = self.total_transactions / uptime
        
        # Calculate average block time
        avg_block_time = statistics.mean(self.block_times) if self.block_times else 0.0
        
        # Calculate average transaction count per block
        avg_tx_per_block = statistics.mean(self.transaction_counts) if self.transaction_counts else 0.0
        
        # Calculate average energy usage per block
        avg_energy_per_block = statistics.mean(self.energy_usage) if self.energy_usage else 0.0
        
        # Calculate average validator confidence
        avg_validator_confidence = statistics.mean(self.validator_confidences) if self.validator_confidences else 0.0
        
        # Calculate average anomaly score
        avg_anomaly_score = statistics.mean(self.anomaly_scores) if self.anomaly_scores else 0.0
        
        return {
            'uptime': uptime,
            'total_blocks': self.total_blocks,
            'total_transactions': self.total_transactions,
            'tps': tps,
            'avg_block_time': avg_block_time,
            'avg_tx_per_block': avg_tx_per_block,
            'avg_energy_per_block': avg_energy_per_block,
            'total_energy': self.total_energy,
            'avg_validator_confidence': avg_validator_confidence,
            'avg_anomaly_score': avg_anomaly_score
        }
    
    def get_performance_improvement(self) -> Dict[str, Any]:
        """
        Calculate performance improvements due to AI components.
        
        Returns:
            Dict[str, Any]: Performance improvement metrics
        """
        # These would be calculated by comparing with baseline metrics
        # For now, we'll just provide some estimates based on the paper's findings
        
        # Efficiency improvement in transaction verification (up to 37%)
        tx_verification_improvement = 0.37
        
        # Fraud detection accuracy improvement (up to 42%)
        fraud_detection_improvement = 0.42
        
        # Energy consumption savings (estimated)
        energy_savings = 0.25  # 25% reduction
        
        return {
            'tx_verification_improvement': tx_verification_improvement,
            'fraud_detection_improvement': fraud_detection_improvement,
            'energy_savings': energy_savings
        } 