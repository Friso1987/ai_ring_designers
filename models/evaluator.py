# models/evaluator.py

import numpy as np
import trimesh
from typing import Dict, Tuple
from dataclasses import dataclass

# DEZE IMPORT ONTBRAK:
from design.parameters import DesignParameters

@dataclass
class DesignMetrics:
    """Alle metrics die we tracken"""
    structural_score: float      # Sterkte
    aesthetic_score: float       # Visuele kwaliteit
    printability_score: float    # Hoe goed te printen
    material_efficiency: float   # Materiaal gebruik
    functional_score: float      # Functionele requirements
    overall_score: float         # Weighted sum
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'structural': self.structural_score,
            'aesthetic': self.aesthetic_score,
            'printability': self.printability_score,
            'material': self.material_efficiency,
            'functional': self.functional_score,
            'overall': self.overall_score,
        }


class DesignEvaluator:
    """Evaluate design quality"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'structural': 0.3,
            'aesthetic': 0.25,
            'printability': 0.2,
            'material': 0.15,
            'functional': 0.1,
        }
    
    def evaluate(self, mesh: trimesh.Trimesh, params: DesignParameters) -> DesignMetrics:
        """Comprehensive evaluation"""
        
        structural = self._evaluate_structural(mesh)
        aesthetic = self._evaluate_aesthetic(mesh, params)
        printability = self._evaluate_printability(mesh)
        material = self._evaluate_material_efficiency(mesh, params)
        functional = self._evaluate_functional(mesh, params)
        
        overall = (
            self.weights['structural'] * structural +
            self.weights['aesthetic'] * aesthetic +
            self.weights['printability'] * printability +
            self.weights['material'] * material +
            self.weights['functional'] * functional
        )
        
        return DesignMetrics(
            structural_score=structural,
            aesthetic_score=aesthetic,
            printability_score=printability,
            material_efficiency=material,
            functional_score=functional,
            overall_score=overall,
        )
    
    def _evaluate_structural(self, mesh: trimesh.Trimesh) -> float:
        """
        Evaluate structural integrity
        - Thickness distribution
        - Weak points
        - Load bearing capacity (simplified FEA)
        """
        score = 0.0
        
        # Check for minimum thickness everywhere
        # (Simplified - in reality run FEA simulation)
        if mesh.is_watertight:
            score += 0.3
        
        # Volume vs surface area ratio
        volume = mesh.volume
        area = mesh.area
        if volume > 0 and area > 0:
            ratio = volume / (area ** 1.5)
            score += min(0.3, ratio * 10)
        
        # Check for overhangs and weak points
        # Analyze face normals
        normals = mesh.face_normals
        vertical_faces = np.abs(normals[:, 2]) < 0.3
        overhang_ratio = np.sum(vertical_faces) / len(normals)
        score += 0.2 * (1 - overhang_ratio)
        
        # Connectivity - no floating parts
        if len(mesh.split()) == 1:
            score += 0.2
        
        return min(1.0, score)
    
    def _evaluate_aesthetic(self, mesh: trimesh.Trimesh, params: DesignParameters) -> float:
        """
        Evaluate visual quality
        - Symmetry
        - Complexity
        - Organic flow
        """
        score = 0.0
        
        # Complexity via edge count (want interesting, not boring)
        edge_count = len(mesh.edges)
        target_complexity = 5000  # Adjust based on your preference
        complexity_score = 1.0 - abs(edge_count - target_complexity) / target_complexity
        score += 0.3 * max(0, complexity_score)
        
        # Surface smoothness via curvature
        try:
            curvature = mesh.vertex_defects
            smoothness = 1.0 - np.std(curvature) / (np.mean(np.abs(curvature)) + 1e-6)
            score += 0.2 * np.clip(smoothness, 0, 1)
        except:
            pass
        
        # Organic variation check
        score += 0.3 * params.organic_variation
        
        # Rotational aesthetics
        # Check if design looks good from multiple angles
        score += 0.2  # Placeholder - zou renders van meerdere hoeken moeten analyseren
        
        return min(1.0, score)
    
    def _evaluate_printability(self, mesh: trimesh.Trimesh) -> float:
        """
        Evaluate 3D printability
        - Overhangs
        - Support requirements
        - Layer adhesion
        """
        score = 1.0
        
        # Check overhangs (faces with normal pointing down)
        normals = mesh.face_normals
        overhang_angle = 45  # degrees
        overhang_threshold = np.cos(np.radians(90 - overhang_angle))
        
        overhangs = normals[:, 2] < -overhang_threshold
        overhang_ratio = np.sum(overhangs) / len(normals)
        score -= 0.4 * overhang_ratio  # Penalty for overhangs
        
        # Check for thin features
        # (Simplified - should analyze wall thickness properly)
        
        # Flatness of bottom
        bottom_vertices = mesh.vertices[mesh.vertices[:, 2] < np.percentile(mesh.vertices[:, 2], 10)]
        if len(bottom_vertices) > 0:
            bottom_flatness = 1.0 - np.std(bottom_vertices[:, 2]) / (np.ptp(bottom_vertices[:, 2]) + 1e-6)
            score += 0.2 * np.clip(bottom_flatness, 0, 1)
        
        # Size check - not too large
        bounds = mesh.bounds
        max_dim = np.max(bounds[1] - bounds[0])
        if max_dim < 200:  # mm
            score += 0.2
        
        return max(0.0, score)
    
    def _evaluate_material_efficiency(self, mesh: trimesh.Trimesh, params: DesignParameters) -> float:
        """
        Evaluate material usage
        - Not too heavy
        - Not too thin/weak
        """
        volume = mesh.volume
        
        # Target volume range (in mm³)
        target_min = 10000
        target_max = 30000
        
        if target_min <= volume <= target_max:
            score = 1.0
        elif volume < target_min:
            score = volume / target_min
        else:
            score = target_max / volume
        
        return np.clip(score, 0, 1)
    
    def _evaluate_functional(self, mesh: trimesh.Trimesh, params: DesignParameters) -> float:
        """
        Evaluate functional requirements
        - Inner diameter correct
        - Gap width correct
        - Mounting points present
        """
        score = 0.0
        
        # Check inner diameter (simplified)
        # In reality: slice mesh and check circular opening
        score += 0.4  # Placeholder
        
        # Check gap presence
        score += 0.3  # Placeholder
        
        # Check mounting points
        score += 0.3  # Placeholder
        
        return score
    
# Voeg deze class toe aan het einde van models/evaluator.py

class MockEvaluator:
    """Fast evaluator zonder mesh rendering - evalueert alleen op parameters"""
    
    def __init__(self):
        self.weights = {
            'complexity': 0.3,
            'balance': 0.25,
            'diversity': 0.25,
            'feasibility': 0.2,
        }
    
    def evaluate(self, mesh, params: DesignParameters) -> DesignMetrics:
        """
        Evaluate design based on parameters only (no mesh needed)
        This is much faster for initial testing
        """
        
        # Complexity score (niet te simpel, niet te complex)
        complexity = self._evaluate_complexity(params)
        
        # Balance score (parameters zijn gebalanceerd)
        balance = self._evaluate_balance(params)
        
        # Diversity score (niet te extreem)
        diversity = self._evaluate_diversity(params)
        
        # Feasibility score (printbaar/realistisch)
        feasibility = self._evaluate_feasibility(params)
        
        overall = (
            self.weights['complexity'] * complexity +
            self.weights['balance'] * balance +
            self.weights['diversity'] * diversity +
            self.weights['feasibility'] * feasibility
        )
        
        return DesignMetrics(
            structural_score=feasibility,
            aesthetic_score=complexity,
            printability_score=feasibility,
            material_efficiency=balance,
            functional_score=1.0,  # Altijd OK want fixed dimensions
            overall_score=overall,
        )
    
    def _evaluate_complexity(self, params: DesignParameters) -> float:
        """Evaluate complexity - want interesting but not too complex"""
        score = 0.0
        
        # Sweet spot for veins: 6-12
        veins_score = 1.0 - abs(params.num_main_veins - 9) / 9.0
        score += 0.3 * max(0, veins_score)
        
        # Sweet spot for layers: 6-12
        layers_score = 1.0 - abs(params.height_layers - 9) / 9.0
        score += 0.3 * max(0, layers_score)
        
        # Branching adds complexity
        score += 0.2 * params.branch_probability
        
        # Wandering adds visual interest
        wander_score = min(1.0, params.lateral_wander / 20.0)
        score += 0.2 * wander_score
        
        return min(1.0, score)
    
    def _evaluate_balance(self, params: DesignParameters) -> float:
        """Evaluate if parameters are balanced"""
        score = 1.0
        
        # Thickness should match complexity
        if params.num_main_veins > 10 and params.vein_thickness < 2.0:
            score -= 0.2  # Te veel dunne veins
        
        if params.num_main_veins < 6 and params.vein_thickness > 4.0:
            score -= 0.2  # Te weinig dikke veins
        
        # Branch length should match branch probability
        if params.branch_probability > 0.7 and params.sub_branch_length < 3:
            score -= 0.2  # Veel korte branches
        
        # Node density should match vein count
        if params.num_main_veins > 10 and params.node_density < 0.3:
            score -= 0.1  # Veel veins maar weinig nodes
        
        return max(0.0, score)
    
    def _evaluate_diversity(self, params: DesignParameters) -> float:
        """Evaluate diversity - not too extreme"""
        score = 1.0
        
        # Penalties for extreme values
        if params.lateral_wander > 20:
            score -= 0.3
        if params.lateral_wander < 7:
            score -= 0.2
        
        if params.organic_variation > 0.8:
            score -= 0.2
        if params.organic_variation < 0.2:
            score -= 0.2
        
        if params.node_density > 0.9 or params.node_density < 0.2:
            score -= 0.2
        
        return max(0.0, score)
    
    def _evaluate_feasibility(self, params: DesignParameters) -> float:
        """Evaluate if design is feasible to print"""
        score = 1.0
        
        # Too thin is bad
        if params.vein_thickness < 1.5:
            score -= 0.3
        
        # Too many branches is hard to print
        if params.branch_probability > 0.8 and params.sub_branch_length > 5:
            score -= 0.2
        
        # Too much variation makes it unpredictable
        if params.organic_variation > 0.7:
            score -= 0.2
        
        return max(0.0, score)
    
class MockEvaluator:
        
    def __init__(self):
            self.weights = {
            'complexity': 0.3,
            'balance': 0.25,
            'diversity': 0.25,
            'feasibility': 0.2,
        }
    
    def evaluate(self, mesh, params) -> DesignMetrics:
        """
        Evaluate design based on parameters only (no mesh needed)
        This is much faster for initial testing
        """
        # Import here to avoid circular imports
        from design.parameters import DesignParameters
        
        # Complexity score (niet te simpel, niet te complex)
        complexity = self._evaluate_complexity(params)
        
        # Balance score (parameters zijn gebalanceerd)
        balance = self._evaluate_balance(params)
        
        # Diversity score (niet te extreem)
        diversity = self._evaluate_diversity(params)
        
        # Feasibility score (printbaar/realistisch)
        feasibility = self._evaluate_feasibility(params)
        
        overall = (
            self.weights['complexity'] * complexity +
            self.weights['balance'] * balance +
            self.weights['diversity'] * diversity +
            self.weights['feasibility'] * feasibility
        )
        
        return DesignMetrics(
            structural_score=feasibility,
            aesthetic_score=complexity,
            printability_score=feasibility,
            material_efficiency=balance,
            functional_score=1.0,  # Altijd OK want fixed dimensions
            overall_score=overall,
        )
    
    def _evaluate_complexity(self, params) -> float:
        """Evaluate complexity - want interesting but not too complex"""
        score = 0.0
        
        # Sweet spot for veins: 6-12
        veins_score = 1.0 - abs(params.num_main_veins - 9) / 9.0
        score += 0.3 * max(0, veins_score)
        
        # Sweet spot for layers: 6-12
        layers_score = 1.0 - abs(params.height_layers - 9) / 9.0
        score += 0.3 * max(0, layers_score)
        
        # Branching adds complexity
        score += 0.2 * params.branch_probability
        
        # Wandering adds visual interest
        wander_score = min(1.0, params.lateral_wander / 20.0)
        score += 0.2 * wander_score
        
        return min(1.0, score)
    
    def _evaluate_balance(self, params) -> float:
        """Evaluate if parameters are balanced"""
        score = 1.0
        
        # Thickness should match complexity
        if params.num_main_veins > 10 and params.vein_thickness < 2.0:
            score -= 0.2  # Te veel dunne veins
        
        if params.num_main_veins < 6 and params.vein_thickness > 4.0:
            score -= 0.2  # Te weinig dikke veins
        
        # Branch length should match branch probability
        if params.branch_probability > 0.7 and params.sub_branch_length < 3:
            score -= 0.2  # Veel korte branches
        
        # Node density should match vein count
        if params.num_main_veins > 10 and params.node_density < 0.3:
            score -= 0.1  # Veel veins maar weinig nodes
        
        return max(0.0, score)
    
    def _evaluate_diversity(self, params) -> float:
        """Evaluate diversity - not too extreme"""
        score = 1.0
        
        # Penalties for extreme values
        if params.lateral_wander > 20:
            score -= 0.3
        if params.lateral_wander < 7:
            score -= 0.2
        
        if params.organic_variation > 0.8:
            score -= 0.2
        if params.organic_variation < 0.2:
            score -= 0.2
        
        if params.node_density > 0.9 or params.node_density < 0.2:
            score -= 0.2
        
        return max(0.0, score)
    
    def _evaluate_feasibility(self, params) -> float:
        """Evaluate if design is feasible to print"""
        score = 1.0
        
        # Too thin is bad
        if params.vein_thickness < 1.5:
            score -= 0.3
        
        # Too many branches is hard to print
        if params.branch_probability > 0.8 and params.sub_branch_length > 5:
            score -= 0.2
        
        # Too much variation makes it unpredictable
        if params.organic_variation > 0.7:
            score -= 0.2
        
        return max(0.0, score)