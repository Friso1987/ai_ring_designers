# models/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path
from datetime import datetime
import logging

from models.generator import DesignGenerator, DesignCritic
from models.evaluator import DesignEvaluator, DesignMetrics
from design.parameters import DesignParameters
from design.scad_generator import SCADGenerator
from design.renderer import DesignRenderer


class ExperienceBuffer:
    """Stores past designs and their scores for learning"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.scores = []
    
    def add(self, params: DesignParameters, metrics: DesignMetrics):
        """Add design to buffer"""
        if len(self.buffer) >= self.max_size:
            # Remove worst design
            worst_idx = np.argmin(self.scores)
            self.buffer.pop(worst_idx)
            self.scores.pop(worst_idx)
        
        self.buffer.append(params)
        self.scores.append(metrics.overall_score)
    
    def sample(self, batch_size: int) -> List[Tuple[DesignParameters, float]]:
        """Sample batch from buffer (weighted by score)"""
        if len(self.buffer) < batch_size:
            return list(zip(self.buffer, self.scores))
        
        # Sample with probability proportional to score
        probs = np.array(self.scores)
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        return [(self.buffer[i], self.scores[i]) for i in indices]
    
    def get_best(self, n: int = 10) -> List[Tuple[DesignParameters, float]]:
        """Get top n designs"""
        sorted_indices = np.argsort(self.scores)[-n:][::-1]
        return [(self.buffer[i], self.scores[i]) for i in sorted_indices]
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        if not self.scores:
            return {}
        
        return {
            'size': len(self.buffer),
            'mean_score': np.mean(self.scores),
            'max_score': np.max(self.scores),
            'min_score': np.min(self.scores),
            'std_score': np.std(self.scores),
        }


class HybridTrainer:
    """
    Combines multiple training approaches:
    1. Reinforcement Learning (Policy Gradient)
    2. Evolutionary Algorithms
    3. GAN-style adversarial training
    """
    
    def __init__(
        self,
        generator: DesignGenerator,
        critic: DesignCritic,
        evaluator: DesignEvaluator,
        scad_generator: SCADGenerator,
        renderer: DesignRenderer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = device
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.evaluator = evaluator
        self.scad_generator = scad_generator
        self.renderer = renderer
        
        # Optimizers
        self.gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
        
        # Experience buffer
        self.buffer = ExperienceBuffer(max_size=5000)
        
        # Training stats
        self.generation = 0
        self.best_score = 0.0
        self.best_design = None
        self.history = {
            'generation': [],
            'mean_score': [],
            'max_score': [],
            'gen_loss': [],
            'critic_loss': [],
        }
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_and_evaluate_batch(
        self, 
        batch_size: int = 16,
        use_diversity: bool = True
    ) -> List[Tuple[DesignParameters, DesignMetrics, torch.Tensor]]:
        """
        Generate batch of designs and evaluate them
        Returns: list of (params, metrics, params_tensor)
        """
        results = []
        
        # Generate designs
        with torch.no_grad():
            z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
            params_tensor = self.generator(z)
        
        # Evaluate each design
        for i in range(batch_size):
            try:
                # Convert to DesignParameters
                params_vec = params_tensor[i].cpu().numpy()
                params = DesignParameters.from_vector(params_vec)
                
                # Generate OpenSCAD code
                scad_code = self.scad_generator.generate(params)
                
                # Render to mesh
                mesh = self.renderer.render_to_mesh(scad_code)
                
                # Evaluate
                metrics = self.evaluator.evaluate(mesh, params)
                
                # Add to buffer
                self.buffer.add(params, metrics)
                
                results.append((params, metrics, params_tensor[i]))
                
                self.logger.info(f"Design {i+1}/{batch_size}: Score={metrics.overall_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating design {i}: {e}")
                continue
        
        return results
    
    def train_step_rl(self, batch_results: List) -> float:
        """
        Reinforcement Learning step (Policy Gradient)
        Maximize expected reward (design score)
        """
        if not batch_results:
            return 0.0
        
        # Extract scores
        scores = torch.tensor([r[1].overall_score for r in batch_results], dtype=torch.float32).to(self.device)
        
        # Normalize scores (advantage)
        if len(scores) > 1:
            advantages = (scores - scores.mean()) / (scores.std() + 1e-8)
        else:
            advantages = scores
        
        # Re-generate parameters through generator to get gradients
        batch_size = len(batch_results)
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        params_tensors = self.generator(z)
        
        # Compute log probability (simplified policy gradient)
        # We use negative MSE as a proxy for log probability
        target_params = torch.stack([r[2] for r in batch_results]).to(self.device).detach()
        log_probs = -torch.sum((params_tensors - target_params)**2, dim=1)
        
        # Policy gradient loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Update generator
        self.gen_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.gen_optimizer.step()
        
        return policy_loss.item()
    
    def train_step_gan(self, batch_results: List) -> Tuple[float, float]:
        """
        GAN-style adversarial training
        Critic learns to distinguish good vs bad designs
        Generator learns to fool critic (make good designs)
        """
        if len(batch_results) < 4:
            return 0.0, 0.0
        
        # Get real (high-scoring) and fake (low-scoring) examples
        scores = [r[1].overall_score for r in batch_results]
        sorted_indices = np.argsort(scores)
        
        n_samples = len(batch_results) // 2
        good_indices = sorted_indices[-n_samples:]
        bad_indices = sorted_indices[:n_samples]
        
        good_params = torch.stack([batch_results[i][2] for i in good_indices]).to(self.device).detach()
        bad_params = torch.stack([batch_results[i][2] for i in bad_indices]).to(self.device).detach()
        
        # Train Critic
        self.critic_optimizer.zero_grad()
        
        good_scores_pred = self.critic(good_params)
        bad_scores_pred = self.critic(bad_params)
        
        # Critic loss: distinguish good from bad
        critic_loss = -(torch.mean(good_scores_pred) - torch.mean(bad_scores_pred))
        
        # Gradient penalty for stability
        alpha = torch.rand(n_samples, 1).to(self.device)
        interpolates = (alpha * good_params + (1 - alpha) * bad_params).requires_grad_(True)
        disc_interpolates = self.critic(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        critic_loss += 10 * gradient_penalty
        
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train Generator (every 5 critic steps)
        gen_loss = 0.0
        if self.generation % 5 == 0:
            self.gen_optimizer.zero_grad()
            
            z = torch.randn(n_samples, self.generator.latent_dim).to(self.device)
            generated_params = self.generator(z)
            gen_scores_pred = self.critic(generated_params)
            
            # Generator loss: fool critic (make it think designs are good)
            gen_loss = -torch.mean(gen_scores_pred)
            
            gen_loss.backward()
            self.gen_optimizer.step()
            
            return gen_loss.item(), critic_loss.item()
        
        return 0.0, critic_loss.item()
    
    def evolutionary_step(self, n_parents: int = 5, n_offspring: int = 20) -> List[DesignParameters]:
        """
        Evolutionary algorithm step
        Take best designs, mutate/crossover, evaluate offspring
        """
        # Get best designs from buffer
        parents = [p for p, s in self.buffer.get_best(n_parents)]
        
        if len(parents) < 2:
            return []
        
        offspring = []
        
        for _ in range(n_offspring):
            # Select two parents
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            
            # Crossover
            vec1 = parent1.to_vector()
            vec2 = parent2.to_vector()
            
            # Uniform crossover
            mask = np.random.random(len(vec1)) > 0.5
            child_vec = np.where(mask, vec1, vec2)
            
            # Mutation
            mutation_mask = np.random.random(len(child_vec)) < 0.2
            noise = np.random.normal(0, 0.15, len(child_vec))
            child_vec = child_vec + mutation_mask * noise * child_vec
            
            child = DesignParameters.from_vector(child_vec)
            offspring.append(child)
        
        return offspring
    
    def train(
        self,
        n_generations: int = 1000,
        batch_size: int = 16,
        eval_interval: int = 10,
        save_interval: int = 50,
        callback=None,
    ):
        """
        Main training loop
        
        callback: function(trainer) called each generation for visualization
        """
        self.logger.info(f"Starting training for {n_generations} generations")
        self.logger.info(f"Device: {self.device}")
        
        for gen in range(n_generations):
            self.generation = gen
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Generation {gen+1}/{n_generations}")
            self.logger.info(f"{'='*60}")
            
            # Phase 1: Generate and evaluate batch
            self.logger.info("Phase 1: Neural network generation")
            nn_results = self.generate_and_evaluate_batch(batch_size)
            
            # Phase 2: Evolutionary step
            if gen > 0 and gen % 5 == 0:
                self.logger.info("Phase 2: Evolutionary step")
                offspring = self.evolutionary_step(n_parents=5, n_offspring=10)
                
                evo_results = []
                for i, params in enumerate(offspring):
                    try:
                        scad_code = self.scad_generator.generate(params)
                        mesh = self.renderer.render_to_mesh(scad_code)
                        metrics = self.evaluator.evaluate(mesh, params)
                        self.buffer.add(params, metrics)

                        params_tensor = torch.tensor(params.to_vector(), dtype=torch.float32).to(self.device)
                        evo_results.append((params, metrics, params_tensor))
                        
                    except Exception as e:
                        self.logger.error(f"Error in evolutionary design {i}: {e}")
                
                nn_results.extend(evo_results)
            
            # Phase 3: Train neural networks
            if nn_results:
                self.logger.info("Phase 3: Training neural networks")
                
                # RL step
                rl_loss = self.train_step_rl(nn_results)
                
                # GAN step
                gen_loss, critic_loss = self.train_step_gan(nn_results)
                
                # Track statistics
                scores = [r[1].overall_score for r in nn_results]
                mean_score = np.mean(scores)
                max_score = np.max(scores)
                
                self.history['generation'].append(gen)
                self.history['mean_score'].append(mean_score)
                self.history['max_score'].append(max_score)
                self.history['gen_loss'].append(gen_loss)
                self.history['critic_loss'].append(critic_loss)
                
                # Update best design
                if max_score > self.best_score:
                    self.best_score = max_score
                    best_idx = np.argmax(scores)
                    self.best_design = nn_results[best_idx][0]
                    
                    self.logger.info(f"🎉 NEW BEST DESIGN! Score: {self.best_score:.3f}")
                    self.save_best_design()
                
                # Log statistics
                buffer_stats = self.buffer.get_stats()
                self.logger.info(f"Batch - Mean: {mean_score:.3f}, Max: {max_score:.3f}")
                self.logger.info(f"Buffer - Mean: {buffer_stats.get('mean_score', 0):.3f}, "
                               f"Max: {buffer_stats.get('max_score', 0):.3f}, "
                               f"Size: {buffer_stats.get('size', 0)}")
                self.logger.info(f"Losses - Gen: {gen_loss:.4f}, Critic: {critic_loss:.4f}, RL: {rl_loss:.4f}")
            
            # Callback for visualization
            if callback:
                callback(self)
            
            # Periodic evaluation
            if (gen + 1) % eval_interval == 0:
                self.evaluate_progress()
            
            # Save checkpoint
            if (gen + 1) % save_interval == 0:
                self.save_checkpoint()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Training completed!")
        self.logger.info(f"Best score achieved: {self.best_score:.3f}")
        self.logger.info("="*60)
    
    def evaluate_progress(self):
        """Evaluate and log current progress"""
        best_designs = self.buffer.get_best(n=5)
        
        self.logger.info("\nTop 5 designs:")
        for i, (params, score) in enumerate(best_designs):
            self.logger.info(f"  {i+1}. Score: {score:.3f} - Veins: {params.num_main_veins}, "
                           f"Layers: {params.height_layers}, Wander: {params.lateral_wander:.1f}")
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = Path('data/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_gen_{self.generation}.pt'
        
        torch.save({
            'generation': self.generation,
            'generator_state': self.generator.state_dict(),
            'critic_state': self.critic.state_dict(),
            'gen_optimizer_state': self.gen_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'best_score': self.best_score,
            'best_design': self.best_design,
            'history': self.history,
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_best_design(self):
        """Save best design separately"""
        if self.best_design is None:
            return
        
        designs_dir = Path('data/designs/best')
        designs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save parameters (convert numpy types to native Python types for JSON)
        params_path = designs_dir / f'best_gen_{self.generation}.json'
        with open(params_path, 'w') as f:
            json.dump({
                'generation': int(self.generation),
                'score': float(self.best_score),
                'parameters': {
                    'num_main_veins': int(self.best_design.num_main_veins),
                    'vein_thickness': float(self.best_design.vein_thickness),
                    'lateral_wander': float(self.best_design.lateral_wander),
                    'radial_wander': float(self.best_design.radial_wander),
                    'height_layers': int(self.best_design.height_layers),
                    'layer_twist_factor': float(self.best_design.layer_twist_factor),
                    'branch_probability': float(self.best_design.branch_probability),
                    'node_density': float(self.best_design.node_density),
                }
            }, f, indent=2)
        
        # Generate and save OpenSCAD
        scad_code = self.scad_generator.generate(self.best_design)
        scad_path = designs_dir / f'best_gen_{self.generation}.scad'
        with open(scad_path, 'w') as f:
            f.write(scad_code)
        
        # Render and save image
        try:
            image = self.renderer.render_to_image(scad_code)
            image_path = designs_dir / f'best_gen_{self.generation}.png'
            image.save(image_path)
        except Exception as e:
            self.logger.error(f"Error rendering image: {e}")