# models/generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DesignGenerator(nn.Module):
    """
    Neural network dat leert goede designs te genereren
    
    Kan werken in 2 modes:
    1. Unconditional: genereer random design
    2. Conditional: genereer design gegeven requirements
    """
    
    def __init__(
        self, 
        latent_dim: int = 64,
        param_dim: int = 14,
        condition_dim: int = 8,
        conditional: bool = False  # BELANGRIJK: default is False nu
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.conditional = conditional
        self.condition_dim = condition_dim
        
        # FIXED: input_dim depends on conditional flag
        if conditional:
            input_dim = latent_dim + condition_dim
            # Encoder voor conditions
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_dim, 32),
                nn.ReLU(),
                nn.Linear(32, condition_dim),
            )
        else:
            input_dim = latent_dim
            self.condition_encoder = None
        
        # Main generator network
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            
            nn.Linear(128, param_dim),
        )
        
        # Parameter-specific output heads voor betere controle
        self.heads = nn.ModuleDict({
            'structure': nn.Linear(param_dim, 7),    # structurele params
            'organic': nn.Linear(param_dim, 4),      # organische params
            'detail': nn.Linear(param_dim, 3),       # detail params
        })
    
    def forward(self, z: torch.Tensor, conditions: torch.Tensor = None):
        """
        z: latent vector [batch_size, latent_dim]
        conditions: optional conditions [batch_size, condition_dim]
        """
        if self.conditional and conditions is not None:
            conditions = self.condition_encoder(conditions)
            x = torch.cat([z, conditions], dim=1)
        else:
            x = z
        
        # Main features
        features = self.generator(x)
        
        # Specialized outputs
        structure = self.heads['structure'](features)
        organic = self.heads['organic'](features)
        detail = self.heads['detail'](features)
        
        # Concatenate
        output = torch.cat([structure, organic, detail], dim=1)
        
        # Apply appropriate activations per parameter type
        output = self._apply_constraints(output)
        
        return output
    
    def _apply_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-specific constraints"""
        result = torch.zeros_like(x)
        
        # Integers (counts) - use sigmoid for smooth gradients
        result[:, 0] = 4 + 12 * torch.sigmoid(x[:, 0])  # num_main_veins: 4-16
        result[:, 4] = 4 + 12 * torch.sigmoid(x[:, 4])  # height_layers: 4-16
        result[:, 9] = 2 + 6 * torch.sigmoid(x[:, 9])   # sub_branch_length: 2-8
        
        # Floats with ranges
        result[:, 1] = 1.0 + 4.0 * torch.sigmoid(x[:, 1])    # vein_thickness: 1-5
        result[:, 2] = 5.0 + 20.0 * torch.sigmoid(x[:, 2])   # lateral_wander: 5-25
        result[:, 3] = 1.0 + 7.0 * torch.sigmoid(x[:, 3])    # radial_wander: 1-8
        result[:, 8] = 10.0 + 50.0 * torch.sigmoid(x[:, 8])  # branch_angle_var: 10-60
        
        # Probabilities/factors (0-1)
        for idx in [5, 6, 7, 10, 11, 12, 13]:
            result[:, idx] = torch.sigmoid(x[:, idx])
        
        return result
    
    def generate(self, batch_size: int = 1, conditions: torch.Tensor = None):
        """Generate new designs"""
        z = torch.randn(batch_size, self.latent_dim)
        with torch.no_grad():
            params = self.forward(z, conditions)
        return params


class DesignCritic(nn.Module):
    """
    Discriminator/Critic voor GAN-style training
    Leert onderscheid maken tussen goede en slechte designs
    """
    
    def __init__(self, param_dim: int = 14):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(param_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 1),
        )
    
    def forward(self, params: torch.Tensor):
        """Return quality score"""
        return self.network(params)