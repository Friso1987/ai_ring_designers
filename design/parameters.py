# design/parameters.py

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DesignParameters:
    """Alle parameters die het model kan leren"""
    
    # Basis geometrie - FIXED VALUES (niet te leren)
    inner_diameter: float = 55.0
    wall_thickness: float = 5.0
    ring_height: float = 20.0
    gap_width: float = 5.0
    lug_width: float = 8.0
    lug_height: float = 10.0
    lug_thickness: float = 6.0
    
    # Organische structuur - LEARNABLE
    num_main_veins: int = 8
    vein_thickness: float = 2.5
    lateral_wander: float = 12.0
    radial_wander: float = 3.0
    
    # Per-layer variatie - LEARNABLE
    height_layers: int = 8
    layer_twist_factor: float = 0.5
    layer_branch_bias: float = 0.3
    
    # Vertakkingen - LEARNABLE
    branch_probability: float = 0.5
    branch_angle_variation: float = 30.0
    sub_branch_length: int = 4
    
    # Nodes - LEARNABLE
    node_density: float = 0.6
    node_size_variation: float = 0.3
    
    # Texture - LEARNABLE
    surface_roughness: float = 0.2
    organic_variation: float = 0.4
    
    def to_vector(self) -> np.ndarray:
        """Convert naar neural network input (alleen learnable params)"""
        return np.array([
            self.num_main_veins,
            self.vein_thickness,
            self.lateral_wander,
            self.radial_wander,
            self.height_layers,
            self.layer_twist_factor,
            self.layer_branch_bias,
            self.branch_probability,
            self.branch_angle_variation,
            self.sub_branch_length,
            self.node_density,
            self.node_size_variation,
            self.surface_roughness,
            self.organic_variation,
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray):
        """Convert van neural network output"""
        return cls(
            # Fixed params blijven default
            inner_diameter=55.0,
            wall_thickness=5.0,
            ring_height=20.0,
            gap_width=5.0,
            lug_width=8.0,
            lug_height=10.0,
            lug_thickness=6.0,
            
            # Learnable params van vector
            num_main_veins=int(max(4, min(16, vec[0]))),
            vein_thickness=max(1.0, min(5.0, vec[1])),
            lateral_wander=max(5.0, min(25.0, vec[2])),
            radial_wander=max(1.0, min(8.0, vec[3])),
            height_layers=int(max(4, min(16, vec[4]))),
            layer_twist_factor=max(0.0, min(1.0, vec[5])),
            layer_branch_bias=max(0.0, min(1.0, vec[6])),
            branch_probability=max(0.0, min(1.0, vec[7])),
            branch_angle_variation=max(10.0, min(60.0, vec[8])),
            sub_branch_length=int(max(2, min(8, vec[9]))),
            node_density=max(0.0, min(1.0, vec[10])),
            node_size_variation=max(0.0, min(1.0, vec[11])),
            surface_roughness=max(0.0, min(1.0, vec[12])),
            organic_variation=max(0.0, min(1.0, vec[13])),
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> 'DesignParameters':
        """Voor genetic algorithm"""
        vec = self.to_vector()
        mask = np.random.random(len(vec)) < mutation_rate
        noise = np.random.normal(0, 0.1, len(vec))
        vec = vec + mask * noise * vec
        return self.from_vector(vec)


class DesignSpace:
    """Definieert de ruimte waarin het model kan exploreren"""
    
    @staticmethod
    def get_bounds() -> List[Tuple[float, float]]:
        """Bounds voor elke parameter"""
        return [
            (4, 16),      # num_main_veins
            (1.0, 5.0),   # vein_thickness
            (5.0, 25.0),  # lateral_wander
            (1.0, 8.0),   # radial_wander
            (4, 16),      # height_layers
            (0.0, 1.0),   # layer_twist_factor
            (0.0, 1.0),   # layer_branch_bias
            (0.0, 1.0),   # branch_probability
            (10.0, 60.0), # branch_angle_variation
            (2, 8),       # sub_branch_length
            (0.0, 1.0),   # node_density
            (0.0, 1.0),   # node_size_variation
            (0.0, 1.0),   # surface_roughness
            (0.0, 1.0),   # organic_variation
        ]
    
    @staticmethod
    def random_sample() -> DesignParameters:
        """Sample random design"""
        bounds = DesignSpace.get_bounds()
        vec = np.array([
            np.random.uniform(low, high)
            for low, high in bounds
        ])
        return DesignParameters.from_vector(vec)