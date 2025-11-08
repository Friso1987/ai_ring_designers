#!/usr/bin/env python3
"""
AI Ring Designer - Main Training Script
Self-learning system for generating organic 3D-printed connectors
"""

import argparse
import logging
from pathlib import Path
import torch
import threading

from models.generator import DesignGenerator, DesignCritic
from design.scad_generator import SCADGenerator


def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/designs/best',
        'data/designs/archive',
        'data/renders',
        'data/checkpoints',
        'logs',
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='AI Ring Designer - Self-learning design system'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=1000,
        help='Number of generations to train (default: 1000)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size per generation (default: 8)'
    )
    
    parser.add_argument(
        '--openscad-path',
        type=str,
        default='openscad',
        help='Path to OpenSCAD executable'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start visualization dashboard'
    )
    
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8050,
        help='Dashboard port (default: 8050)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock renderer (fast, for testing without OpenSCAD)'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("🧬 AI Ring Designer - Self-Learning System")
    print("="*60)
    print(f"Mode: {'🎭 MOCK (Fast Testing)' if args.mock else '🔧 REAL (with OpenSCAD)'}")
    print(f"Device: {device}")
    print(f"Generations: {args.generations}")
    print(f"Batch size: {args.batch_size}")
    if not args.mock:
        print(f"OpenSCAD: {args.openscad_path}")
    print("="*60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    
    generator = DesignGenerator(
        latent_dim=64,
        param_dim=14,
        conditional=False
    )
    
    critic = DesignCritic(param_dim=14)
    
    # MOCK MODE OR REAL MODE
    if args.mock:
        print("   🎭 Using MOCK mode (fast testing, no OpenSCAD needed)")
        from design.mock_renderer import MockRenderer
        from models.evaluator import MockEvaluator
        
        renderer = MockRenderer()
        evaluator = MockEvaluator()
    else:
        print("   🔧 Using REAL mode (with OpenSCAD rendering)")
        from design.renderer import DesignRenderer
        from models.evaluator import DesignEvaluator
        
        renderer = DesignRenderer(openscad_path=args.openscad_path)
        evaluator = DesignEvaluator()
    
    scad_generator = SCADGenerator()
    
    # Import trainer after renderer/evaluator are initialized
    from models.trainer import HybridTrainer
    
    trainer = HybridTrainer(
        generator=generator,
        critic=critic,
        evaluator=evaluator,
        scad_generator=scad_generator,
        renderer=renderer,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Loading checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            generator.load_state_dict(checkpoint['generator_state'])
            critic.load_state_dict(checkpoint['critic_state'])
            trainer.generation = checkpoint['generation']
            trainer.best_score = checkpoint['best_score']
            trainer.best_design = checkpoint['best_design']
            trainer.history = checkpoint['history']
            print(f"✅ Resumed from generation {trainer.generation}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Starting from scratch instead...")
    
    # Start dashboard in separate thread if requested
    if args.dashboard:
        print(f"\n🖥️  Starting dashboard on http://localhost:{args.dashboard_port}")
        try:
            from visualization.dashboard import TrainingDashboard
            dashboard = TrainingDashboard(trainer=trainer, port=args.dashboard_port)
            dashboard_thread = threading.Thread(target=dashboard.run, daemon=True)
            dashboard_thread.start()
            print("✅ Dashboard started")
            print(f"   Open your browser to: http://localhost:{args.dashboard_port}")
        except Exception as e:
            print(f"⚠️  Warning: Could not start dashboard: {e}")
            print("   Continuing without dashboard...")
    
    # Start training
    print("\n🚀 Starting training...\n")
    
    try:
        trainer.train(
            n_generations=args.generations,
            batch_size=args.batch_size,
            eval_interval=10,
            save_interval=50,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Saving checkpoint...")
        trainer.save_checkpoint()
        print("✅ Checkpoint saved")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\n💾 Attempting to save checkpoint...")
        try:
            trainer.save_checkpoint()
            print("✅ Emergency checkpoint saved")
        except:
            print("❌ Could not save checkpoint")
    
    print("\n" + "="*60)
    print("✨ Training completed!")
    print(f"Best score achieved: {trainer.best_score:.3f}")
    print(f"Total designs evaluated: {trainer.buffer.get_stats().get('size', 0)}")
    print("\n📁 Results saved in:")
    print("   - Best designs: data/designs/best/")
    print("   - Checkpoints: data/checkpoints/")
    print("   - Logs: logs/")
    
    if trainer.best_design:
        print(f"\n🏆 Best design parameters:")
        print(f"   Main veins: {trainer.best_design.num_main_veins}")
        print(f"   Layers: {trainer.best_design.height_layers}")
        print(f"   Lateral wander: {trainer.best_design.lateral_wander:.1f}")
        print(f"   Branch probability: {trainer.best_design.branch_probability:.2f}")
    
    print("="*60)


if __name__ == "__main__":
    main()