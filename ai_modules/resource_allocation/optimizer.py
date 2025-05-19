#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resource Allocation AI module using reinforcement learning.

This module optimizes the distribution of computational resources across the network
based on transaction patterns and node performance.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional
import random
import copy
import json
import os

logger = logging.getLogger(__name__)

class ResourceState:
    """Represents the current state of resources in the network."""
    
    def __init__(self):
        """Initialize the resource state."""
        self.node_states = {}
        self.transaction_queue_lengths = {}
        self.historical_performance = {}
        self.network_latencies = {}
        self.last_update_time = time.time()
    
    def update_node_state(self, node_id: str, metrics: Dict[str, float]) -> None:
        """
        Update the state of a node.
        
        Args:
            node_id: ID of the node
            metrics: Resource metrics for the node
        """
        self.node_states[node_id] = metrics
    
    def update_queue_length(self, node_id: str, queue_length: int) -> None:
        """
        Update the transaction queue length for a node.
        
        Args:
            node_id: ID of the node
            queue_length: Length of the transaction queue
        """
        self.transaction_queue_lengths[node_id] = queue_length
    
    def update_performance(self, node_id: str, metrics: Dict[str, float]) -> None:
        """
        Update the historical performance metrics for a node.
        
        Args:
            node_id: ID of the node
            metrics: Performance metrics for the node
        """
        if node_id not in self.historical_performance:
            self.historical_performance[node_id] = []
        
        # Add timestamp to metrics
        metrics['timestamp'] = time.time()
        
        # Add to history
        self.historical_performance[node_id].append(metrics)
        
        # Limit history length
        if len(self.historical_performance[node_id]) > 100:
            self.historical_performance[node_id] = self.historical_performance[node_id][-100:]
    
    def update_network_latency(self, node_id1: str, node_id2: str, latency_ms: float) -> None:
        """
        Update the network latency between two nodes.
        
        Args:
            node_id1: ID of the first node
            node_id2: ID of the second node
            latency_ms: Network latency in milliseconds
        """
        key = f"{node_id1}-{node_id2}"
        self.network_latencies[key] = latency_ms
    
    def get_observation_vector(self, node_id: str) -> List[float]:
        """
        Get an observation vector for a specific node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List[float]: Observation vector for the node
        """
        # Start with basic node metrics
        node_metrics = self.node_states.get(node_id, {})
        cpu = node_metrics.get('cpu', 0.0)
        memory = node_metrics.get('memory', 0.0)
        network_in = node_metrics.get('network_in', 0.0)
        network_out = node_metrics.get('network_out', 0.0)
        
        # Get queue length
        queue_length = self.transaction_queue_lengths.get(node_id, 0)
        normalized_queue = min(queue_length / 10000, 1.0)  # Normalize queue length
        
        # Get average performance metrics
        node_history = self.historical_performance.get(node_id, [])
        if node_history:
            recent_history = node_history[-10:]  # Use last 10 entries
            avg_tps = sum(entry.get('tps', 0.0) for entry in recent_history) / len(recent_history)
            avg_latency = sum(entry.get('latency', 0.0) for entry in recent_history) / len(recent_history)
            normalized_tps = min(avg_tps / 1000, 1.0)  # Normalize TPS
            normalized_latency = min(avg_latency / 1000, 1.0)  # Normalize latency
        else:
            normalized_tps = 0.0
            normalized_latency = 0.0
        
        # Get average network latency
        node_latencies = []
        for key, latency in self.network_latencies.items():
            if key.startswith(f"{node_id}-") or key.endswith(f"-{node_id}"):
                node_latencies.append(latency)
        
        avg_network_latency = 0.0
        if node_latencies:
            avg_network_latency = sum(node_latencies) / len(node_latencies)
        normalized_network_latency = min(avg_network_latency / 1000, 1.0)  # Normalize network latency
        
        # Combine into an observation vector
        observation = [
            cpu,
            memory,
            network_in,
            network_out,
            normalized_queue,
            normalized_tps,
            normalized_latency,
            normalized_network_latency
        ]
        
        return observation

class PPOAgent:
    """Proximal Policy Optimization agent for resource allocation."""
    
    def __init__(self, 
                learning_rate: float = 0.0003, 
                discount_factor: float = 0.99,
                entropy_coef: float = 0.01,
                clip_range: float = 0.2,
                value_coef: float = 0.5):
        """
        Initialize the PPO agent.
        
        Args:
            learning_rate: Learning rate for policy update
            discount_factor: Reward discount factor
            entropy_coef: Entropy coefficient for exploration
            clip_range: Policy clipping range
            value_coef: Value loss coefficient
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        self.clip_range = clip_range
        self.value_coef = value_coef
        
        # In a real implementation, these would be neural networks
        # Here we'll simulate their behavior
        self.policy_network = DummyPolicyNetwork()
        self.value_network = DummyValueNetwork()
        
        # Training data
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        logger.info("PPO agent initialized")
    
    def get_action(self, observation: List[float]) -> Dict[str, float]:
        """
        Get an action based on the current observation.
        
        Args:
            observation: Current observation vector
            
        Returns:
            Dict[str, float]: Resource allocation action
        """
        # Convert observation to tensor (would be done in a real implementation)
        
        # Get action distribution and sample an action
        action, log_prob, value = self.policy_network.forward(observation)
        
        # Store for training
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action
    
    def update(self, final_reward: float) -> Dict[str, float]:
        """
        Update the agent with the final reward.
        
        Args:
            final_reward: Final reward for the episode
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # In a real implementation, this would update the neural networks
        # Here we'll just simulate the update
        
        # Add the final reward
        self.rewards.append(final_reward)
        
        # Calculate advantages and returns
        # This is a simplified version
        
        # Update networks (simulated)
        if random.random() < 0.1:  # Simulate improvement 10% of the time
            self.policy_network.improve()
            self.value_network.improve()
        
        # Clear training data
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # Return metrics
        return {
            'policy_loss': random.uniform(0.1, 0.5),
            'value_loss': random.uniform(0.1, 0.5),
            'entropy': random.uniform(0.01, 0.1),
            'learning_rate': self.learning_rate
        }
    
    def save(self, path: str) -> None:
        """
        Save the agent to a file.
        
        Args:
            path: Path to save the agent
        """
        # In a real implementation, this would save the neural networks
        # Here we'll just save some dummy data
        data = {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'entropy_coef': self.entropy_coef,
            'clip_range': self.clip_range,
            'value_coef': self.value_coef,
            'policy_network_version': self.policy_network.version,
            'value_network_version': self.value_network.version
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved agent to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        # In a real implementation, this would load the neural networks
        # Here we'll just load some dummy data
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.learning_rate = data.get('learning_rate', self.learning_rate)
            self.discount_factor = data.get('discount_factor', self.discount_factor)
            self.entropy_coef = data.get('entropy_coef', self.entropy_coef)
            self.clip_range = data.get('clip_range', self.clip_range)
            self.value_coef = data.get('value_coef', self.value_coef)
            
            # Set network versions
            self.policy_network.version = data.get('policy_network_version', 0)
            self.value_network.version = data.get('value_network_version', 0)
            
            logger.info(f"Loaded agent from {path}")
        else:
            logger.warning(f"Agent file {path} not found")

class DummyPolicyNetwork:
    """Dummy policy network for simulation."""
    
    def __init__(self):
        """Initialize the dummy policy network."""
        self.version = 0
    
    def forward(self, observation: List[float]) -> tuple:
        """
        Forward pass through the network.
        
        Args:
            observation: Observation vector
            
        Returns:
            tuple: Action, log probability, and value
        """
        # Simulate action selection
        cpu_alloc = min(max(observation[0] + random.uniform(-0.1, 0.2), 0.1), 1.0)
        memory_alloc = min(max(observation[1] + random.uniform(-0.1, 0.2), 0.1), 1.0)
        network_alloc = min(max((observation[2] + observation[3]) / 2 + random.uniform(-0.1, 0.2), 0.1), 1.0)
        
        action = {
            'cpu_allocation': cpu_alloc,
            'memory_allocation': memory_alloc,
            'network_allocation': network_alloc
        }
        
        # Simulate log probability
        log_prob = random.uniform(-1.0, 0.0)
        
        # Simulate value
        value = random.uniform(0.0, 1.0)
        
        return action, log_prob, value
    
    def improve(self) -> None:
        """Simulate improvement in the network."""
        self.version += 1

class DummyValueNetwork:
    """Dummy value network for simulation."""
    
    def __init__(self):
        """Initialize the dummy value network."""
        self.version = 0
    
    def forward(self, observation: List[float]) -> float:
        """
        Forward pass through the network.
        
        Args:
            observation: Observation vector
            
        Returns:
            float: Value estimate
        """
        # Simulate value estimation
        return random.uniform(0.0, 1.0)
    
    def improve(self) -> None:
        """Simulate improvement in the network."""
        self.version += 1

class ResourceOptimizer:
    """
    Resource optimizer using reinforcement learning for resource allocation.
    """
    
    def __init__(self, 
                learning_rate: float = 0.0003, 
                discount_factor: float = 0.99,
                update_interval: int = 10000):
        """
        Initialize the resource optimizer.
        
        Args:
            learning_rate: Learning rate for the optimization algorithm
            discount_factor: Discount factor for future rewards
            update_interval: Interval (in steps) for updating the optimizer
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.update_interval = update_interval
        self.step_counter = 0
        
        # Initialize resource state
        self.state = ResourceState()
        
        # Initialize RL agent
        self.agent = PPOAgent(learning_rate, discount_factor)
        
        # Current allocations
        self.current_allocations = {}
        
        # Performance history
        self.performance_history = []
        
        logger.info("Resource optimizer initialized")
    
    def optimize(self, 
               resource_states: Dict[str, Dict[str, float]], 
               transaction_pool, 
               network_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Optimize resource allocation based on current state.
        
        Args:
            resource_states: Current resource usage for each node
            transaction_pool: Current transaction pool
            network_metrics: Current network performance metrics
            
        Returns:
            Dict[str, Dict[str, float]]: Optimized resource allocations for each node
        """
        # Update resource state
        for node_id, metrics in resource_states.items():
            self.state.update_node_state(node_id, metrics)
        
        # Update queue lengths
        for node_id, node_metrics in resource_states.items():
            # In a real implementation, this would be the actual queue length
            # Here we'll estimate based on transaction pool size
            queue_length = len(transaction_pool) // len(resource_states)
            self.state.update_queue_length(node_id, queue_length)
        
        # Update performance metrics
        for node_id, node_metrics in resource_states.items():
            # In a real implementation, this would be actual performance metrics
            # Here we'll estimate based on resource usage and transaction pool
            tps = 100.0 * (1.0 - node_metrics.get('cpu', 0.0)) * (len(resource_states) / max(1, len(transaction_pool) // 100))
            latency = 50.0 + 200.0 * node_metrics.get('cpu', 0.0)
            
            self.state.update_performance(node_id, {
                'tps': tps,
                'latency': latency
            })
        
        # Generate allocations for each node
        allocations = {}
        for node_id in resource_states.keys():
            # Get observation vector for the node
            observation = self.state.get_observation_vector(node_id)
            
            # Get action from agent
            action = self.agent.get_action(observation)
            
            # Store allocation
            allocations[node_id] = action
        
        # Update current allocations
        self.current_allocations = copy.deepcopy(allocations)
        
        # Increment step counter
        self.step_counter += 1
        
        # Update agent
        if self.step_counter % self.update_interval == 0:
            # In a real implementation, this would be the actual reward
            # Here we'll estimate based on network metrics
            reward = network_metrics.get('tps', 0.0) / 1000.0 - network_metrics.get('latency', 0.0) / 1000.0
            
            self.agent.update(reward)
            
            # Add to performance history
            self.performance_history.append({
                'step': self.step_counter,
                'reward': reward,
                'tps': network_metrics.get('tps', 0.0),
                'latency': network_metrics.get('latency', 0.0)
            })
            
            # Limit history length
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            logger.info(f"Optimizer updated at step {self.step_counter}, reward: {reward:.4f}")
        
        return allocations
    
    def save(self, path: str) -> None:
        """
        Save the optimizer state to a file.
        
        Args:
            path: Path to save the optimizer state
        """
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the optimizer state from a file.
        
        Args:
            path: Path to load the optimizer state from
        """
        self.agent.load(path)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the optimizer.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not self.performance_history:
            return {
                'average_reward': 0.0,
                'average_tps': 0.0,
                'average_latency': 0.0
            }
        
        recent_history = self.performance_history[-10:]  # Use last 10 entries
        average_reward = sum(entry['reward'] for entry in recent_history) / len(recent_history)
        average_tps = sum(entry['tps'] for entry in recent_history) / len(recent_history)
        average_latency = sum(entry['latency'] for entry in recent_history) / len(recent_history)
        
        return {
            'average_reward': average_reward,
            'average_tps': average_tps,
            'average_latency': average_latency
        } 