import torch
import torch.nn as nn
import logging

class DEQWrapper(nn.Module):
    """
    A Deep Equilibrium wrapper that can be applied to any module to add
    fixed-point iteration capabilities.
    """
    def __init__(self, module, fixed_point_fn=None, name="unnamed", config=None):
        super().__init__()
        self.module = module
        self.fixed_point_fn = fixed_point_fn or self._default_fixed_point
        
        # Set default values
        self.iterations = 1  # Default iterations
        self.tolerance = 1e-4  # Default convergence tolerance
        self.name = name
        self.track_residuals = False
        self.residual_history = []
        
        # Update from config if provided
        if config is not None and hasattr(config, 'deq'):
            self.iterations = getattr(config.deq, 'default_iterations', 1)
            self.tolerance = getattr(config.deq, 'tolerance', 1e-4)
            self.track_residuals = getattr(config.deq, 'track_residuals', False)
        
    def _default_fixed_point_with_residual(self, z, *args, **kwargs):
        z_new = self.module(z, *args, **kwargs)
        return z_new, torch.norm(z_new - z) / (torch.norm(z) + 1e-8)
        
    def _default_fixed_point(self, z, *args, **kwargs):
        return self.module(z, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        z = x  # Start without cloning to match original behavior
        self.residual_history = []
        actual_iters = 0
        
        for i in range(self.iterations):
            actual_iters += 1
            
            # Compute new state
            z_new = self.fixed_point_fn(z, *args, **kwargs)
            
            # Calculate residual
            residual = torch.norm(z_new - z) / (torch.norm(z) + 1e-8)
            
            if self.track_residuals:
                self.residual_history.append(residual.item())
            
            # Check for convergence
            if residual < self.tolerance:
                z = z_new  # Update to final state
                break
                
            # Update state
            z = z_new
            
        return z, actual_iters

class DEQAttentionBlock(nn.Module):
    """
    Specific implementation of DEQ for the attention + GCN block in DiffPose.
    """
    def __init__(self, attention_layer, gcn_layer, name="unnamed", config=None):
        super().__init__()
        self.attention_layer = attention_layer
        self.gcn_layer = gcn_layer
        
        # Set default values
        self.iterations = 1
        self.tolerance = 1e-4
        self.name = name
        self.residual_history = []
        self.track_residuals = False
        
        # Update from config if provided
        if config is not None and hasattr(config, 'deq'):
            self.iterations = getattr(config.deq, 'default_iterations', 1)
            self.tolerance = getattr(config.deq, 'tolerance', 1e-4)
            self.track_residuals = getattr(config.deq, 'track_residuals', False)
        
    def forward(self, x, mask, temb=None):
        """
        Forward pass with Deep Equilibrium iterations.
        
        Args:
            x: Input tensor
            mask: Attention mask
            temb: Optional time embedding for diffusion
            
        Returns:
            Tuple of (output tensor, number of iterations)
        """
        # If iterations = 1, just do a single standard forward pass (no DEQ)
        if self.iterations <= 1:
            z_attn = self.attention_layer(x, mask)
            if temb is not None:
                z = self.gcn_layer(z_attn, temb)
            else:
                z = self.gcn_layer(z_attn)
            return z, 1
            
        # Initialize for DEQ iterations - start with the input
        z = x  # No clone to match original behavior
        self.residual_history = []
        actual_iters = 0
        
        # Perform DEQ iterations
        for i in range(self.iterations):
            actual_iters += 1
            
            # Compute attention output
            z_attn = self.attention_layer(z, mask)
            
            # Apply GCN layer
            if temb is not None:
                z_new = self.gcn_layer(z_attn, temb)
            else:
                z_new = self.gcn_layer(z_attn)
            
            # Calculate residual
            residual = torch.norm(z_new - z) / (torch.norm(z) + 1e-8)
            
            if self.track_residuals:
                self.residual_history.append(residual.item())
            
            # Check for convergence
            if residual < self.tolerance:
                z = z_new  # Update to final state
                break
                
            # Update state
            z = z_new
            actual_iters += 1
        
        return z, actual_iters

class DEQManager:
    """
    Utility class to manage multiple DEQ components in a model.
    """
    def __init__(self, config=None):
        self.components = []
        self.total_iterations = 0
        self.sample_count = 0
        self.max_iterations = 0
        self.min_iterations = float('inf')
        self.log_frequency = 100  # Log every 100 batches by default
        
        # Configure from config if provided
        if config is not None and hasattr(config, 'deq'):
            self.log_frequency = getattr(config.deq, 'log_frequency', 100)
    
    def register(self, component):
        """Register a DEQ component to be managed."""
        self.components.append(component)
    
    def set_iterations(self, iterations):
        """Set iterations for all registered components."""
        for component in self.components:
            component.iterations = iterations
    
    def set_tolerance(self, tolerance):
        """Set tolerance for all registered components."""
        for component in self.components:
            component.tolerance = tolerance
    
    def reset_stats(self):
        """Reset iteration statistics."""
        self.total_iterations = 0
        self.sample_count = 0
        self.max_iterations = 0
        self.min_iterations = float('inf')
    
    def update_stats(self, iterations):
        """Update iteration statistics."""
        self.total_iterations += iterations
        self.sample_count += 1
        self.max_iterations = max(self.max_iterations, iterations)
        if iterations > 0:  # Avoid counting skipped components
            self.min_iterations = min(self.min_iterations, iterations)
    
    def get_avg_iterations(self):
        """Calculate average iterations per sample."""
        if self.sample_count == 0:
            return 0
        return self.total_iterations / self.sample_count
    
    def log_stats(self):
        """Log iteration statistics."""
        avg = self.get_avg_iterations()
        if avg > 0:
            min_iters = self.min_iterations if self.min_iterations != float('inf') else 0
            logging.info(f"Implicit Iter: Avg {avg:.1f}, Max {self.max_iterations}, Min {min_iters}, Samples {self.sample_count}")
            
    def enable_tracking(self, enable=True):
        """Enable or disable residual tracking for all components."""
        for component in self.components:
            component.track_residuals = enable