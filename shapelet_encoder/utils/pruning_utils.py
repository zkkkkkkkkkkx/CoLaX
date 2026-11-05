"""
Model Pruning Utilities for ProtoPTST Hybrid Model
Supports both structured and unstructured pruning to reduce model parameters.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Returns:
        Dictionary with 'total', 'trainable', and 'non_trainable' parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Get all prunable layers (Linear, Conv1d, Conv2d) from the model.
    
    Returns:
        List of (name, module) tuples for prunable layers
    """
    prunable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            prunable_layers.append((name, module))
    return prunable_layers


def unstructured_prune_layer(
    module: nn.Module,
    pruning_method: str = 'l1_unstructured',
    amount: float = 0.2,
    dim: Optional[int] = None
) -> None:
    """
    Apply unstructured pruning to a single layer.
    
    Args:
        module: The module to prune
        pruning_method: 'l1_unstructured', 'l2_unstructured', 'random_unstructured'
        amount: Fraction of parameters to prune (0.0 to 1.0)
        dim: Dimension to prune along (for structured pruning)
    """
    if isinstance(module, nn.Linear):
        if pruning_method == 'l1_unstructured':
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif pruning_method == 'l2_unstructured':
            prune.l2_unstructured(module, name='weight', amount=amount)
        elif pruning_method == 'random_unstructured':
            prune.random_unstructured(module, name='weight', amount=amount)
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")
    elif isinstance(module, nn.Conv1d):
        if pruning_method == 'l1_unstructured':
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif pruning_method == 'l2_unstructured':
            prune.l2_unstructured(module, name='weight', amount=amount)
        elif pruning_method == 'random_unstructured':
            prune.random_unstructured(module, name='weight', amount=amount)
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")


def structured_prune_layer(
    module: nn.Module,
    pruning_method: str = 'ln_structured',
    amount: float = 0.2,
    dim: int = 0,
    n: int = 2
) -> None:
    """
    Apply structured pruning to a single layer.
    
    Args:
        module: The module to prune
        pruning_method: 'ln_structured' (L-n norm structured pruning)
        amount: Fraction of channels/filters to prune (0.0 to 1.0)
        dim: Dimension to prune along (0 for output channels, 1 for input channels)
        n: Norm to use for ln_structured pruning (default: 2 for L2)
    """
    if isinstance(module, nn.Linear):
        if pruning_method == 'ln_structured':
            prune.ln_structured(module, name='weight', amount=amount, dim=dim, n=n)
        else:
            raise ValueError(f"Unknown structured pruning method: {pruning_method}")
    elif isinstance(module, nn.Conv1d):
        if pruning_method == 'ln_structured':
            prune.ln_structured(module, name='weight', amount=amount, dim=0, n=n)
        else:
            raise ValueError(f"Unknown structured pruning method: {pruning_method}")


def magnitude_based_pruning(
    model: nn.Module,
    amount: float = 0.2,
    exclude_layers: Optional[List[str]] = None,
    structured: bool = False
) -> Dict[str, float]:
    """
    Apply magnitude-based pruning to the model.
    
    Args:
        model: The model to prune
        amount: Fraction of parameters to prune (0.0 to 1.0)
        exclude_layers: List of layer names to exclude from pruning
        structured: If True, use structured pruning; otherwise unstructured
        
    Returns:
        Dictionary with pruning statistics
    """
    if exclude_layers is None:
        exclude_layers = []
    
    prunable_layers = get_prunable_layers(model)
    pruned_layers = []
    
    for name, module in prunable_layers:
        if any(excluded in name for excluded in exclude_layers):
            continue
        
        try:
            if structured:
                structured_prune_layer(module, amount=amount)
            else:
                unstructured_prune_layer(module, pruning_method='l1_unstructured', amount=amount)
            pruned_layers.append(name)
        except Exception as e:
            print(f"Warning: Could not prune layer {name}: {e}")
    
    # Count parameters after pruning
    param_counts = count_parameters(model)
    
    return {
        'pruned_layers': pruned_layers,
        'num_pruned_layers': len(pruned_layers),
        'total_params': param_counts['total'],
        'trainable_params': param_counts['trainable']
    }


def global_magnitude_pruning(
    model: nn.Module,
    amount: float = 0.2,
    exclude_layers: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Apply global magnitude-based pruning across all layers.
    This prunes the smallest magnitude weights globally, not per-layer.
    
    Args:
        model: The model to prune
        amount: Fraction of parameters to prune globally (0.0 to 1.0)
        exclude_layers: List of layer names to exclude from pruning
        
    Returns:
        Dictionary with pruning statistics
    """
    if exclude_layers is None:
        exclude_layers = []
    
    # Collect all weights
    all_weights = []
    weight_locations = []
    
    for name, module in model.named_modules():
        if any(excluded in name for excluded in exclude_layers):
            continue
        
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data.clone()
                all_weights.append(weight.flatten())
                weight_locations.append((name, module, weight.shape))
    
    if not all_weights:
        return {'error': 'No prunable layers found'}
    
    # Concatenate all weights
    all_weights_tensor = torch.cat(all_weights)
    
    # Calculate global threshold
    num_params_to_prune = int(amount * len(all_weights_tensor))
    threshold = torch.kthvalue(torch.abs(all_weights_tensor), num_params_to_prune)[0]
    
    # Prune each layer
    pruned_layers = []
    for name, module, shape in weight_locations:
        mask = torch.abs(module.weight.data) > threshold
        if mask.any():
            prune.custom_from_mask(module, name='weight', mask=mask)
            pruned_layers.append(name)
    
    param_counts = count_parameters(model)
    
    return {
        'pruned_layers': pruned_layers,
        'num_pruned_layers': len(pruned_layers),
        'total_params': param_counts['total'],
        'trainable_params': param_counts['trainable'],
        'threshold': threshold.item()
    }


def remove_pruning_masks(model: nn.Module) -> None:
    """
    Remove pruning masks and make pruning permanent.
    This actually removes the pruned parameters from the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            try:
                prune.remove(module, 'weight')
            except:
                pass


def apply_pruning_to_model(
    model: nn.Module,
    pruning_config: Dict[str, any]
) -> Dict[str, any]:
    """
    Apply pruning to model based on configuration.
    
    Args:
        model: The model to prune
        pruning_config: Dictionary with pruning configuration:
            - 'method': 'magnitude', 'global_magnitude', 'structured'
            - 'amount': Fraction to prune (0.0 to 1.0)
            - 'exclude_layers': List of layer names to exclude
            - 'structured': Whether to use structured pruning
            
    Returns:
        Dictionary with pruning statistics
    """
    method = pruning_config.get('method', 'magnitude')
    amount = pruning_config.get('amount', 0.2)
    exclude_layers = pruning_config.get('exclude_layers', [])
    structured = pruning_config.get('structured', False)
    
    print(f"Applying {method} pruning with amount={amount}")
    print(f"Excluding layers: {exclude_layers}")
    
    if method == 'magnitude':
        stats = magnitude_based_pruning(model, amount=amount, exclude_layers=exclude_layers, structured=structured)
    elif method == 'global_magnitude':
        stats = global_magnitude_pruning(model, amount=amount, exclude_layers=exclude_layers)
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    return stats


def gradual_pruning(
    model: nn.Module,
    initial_sparsity: float,
    final_sparsity: float,
    num_steps: int,
    exclude_layers: Optional[List[str]] = None,
    structured: bool = False
) -> List[Dict[str, any]]:
    """
    Gradually increase pruning sparsity over multiple steps.
    Useful for training with pruning.
    
    Args:
        model: The model to prune
        initial_sparsity: Initial sparsity (0.0 to 1.0)
        final_sparsity: Final sparsity (0.0 to 1.0)
        num_steps: Number of pruning steps
        exclude_layers: List of layer names to exclude
        structured: Whether to use structured pruning
        
    Returns:
        List of pruning statistics for each step
    """
    sparsity_schedule = np.linspace(initial_sparsity, final_sparsity, num_steps)
    all_stats = []
    
    for step, sparsity in enumerate(sparsity_schedule):
        print(f"Pruning step {step+1}/{num_steps}, sparsity: {sparsity:.3f}")
        stats = magnitude_based_pruning(
            model, 
            amount=sparsity, 
            exclude_layers=exclude_layers,
            structured=structured
        )
        stats['step'] = step
        stats['sparsity'] = sparsity
        all_stats.append(stats)
    
    return all_stats











