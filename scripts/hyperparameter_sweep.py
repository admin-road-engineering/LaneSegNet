#!/usr/bin/env python3
"""
Hyperparameter Sweep Framework for ViT-Base Optimization.
Systematic grid search to find optimal training configuration before architectural changes.
"""

import os
import sys
import json
import time
import subprocess
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterSweep:
    """
    Framework for systematic hyperparameter optimization.
    """
    
    def __init__(self, base_save_dir="work_dirs/hyperparameter_sweep"):
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        # Create sweep timestamp
        self.sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = self.base_save_dir / f"sweep_{self.sweep_id}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Hyperparameter sweep initialized: {self.sweep_dir}")
    
    def define_search_space(self):
        """
        Define hyperparameter search space based on Phase 4A priorities.
        """
        # PRIORITY 1: Learning rate optimization (highest impact)
        learning_rates = {
            'encoder_lr': [5e-6, 1e-5, 2e-5, 5e-5],  # Lower for pre-trained
            'decoder_lr': [1e-4, 5e-4, 1e-3, 2e-3]   # Higher for new layers
        }
        
        # PRIORITY 2: Scheduler optimization (convergence quality)
        schedulers = [
            {'scheduler': 'plateau', 'scheduler_factor': 0.5, 'scheduler_patience': 5},
            {'scheduler': 'plateau', 'scheduler_factor': 0.3, 'scheduler_patience': 3},
            {'scheduler': 'cosine', 'scheduler_factor': 1.0},  # T_max = epochs
            {'scheduler': 'cosine-warmup', 'scheduler_factor': 1.0, 'warmup_steps': 100},  # With warmup
            {'scheduler': 'cosine-warmup', 'scheduler_factor': 1.0, 'warmup_steps': 500},  # More warmup
        ]
        
        # PRIORITY 3: Optimizer configuration (stability)
        optimizers = [
            {'optimizer': 'adamw', 'weight_decay': 1e-4, 'beta1': 0.9, 'beta2': 0.999},
            {'optimizer': 'adamw', 'weight_decay': 5e-5, 'beta1': 0.9, 'beta2': 0.999},
            {'optimizer': 'adamw', 'weight_decay': 1e-4, 'beta1': 0.95, 'beta2': 0.999},
        ]
        
        # PRIORITY 4: Training duration (find optimal stopping point)
        training_configs = [
            {'epochs': 100, 'early_stopping_patience': 15},
            {'epochs': 150, 'early_stopping_patience': 20},
            {'epochs': 200, 'early_stopping_patience': 25},
        ]
        
        # PRIORITY 5: Loss function tuning (class imbalance handling)
        loss_configs = [
            {'ohem_alpha': 0.25, 'ohem_gamma': 2.0, 'dice_weight': 0.6, 'focal_weight': 0.4},
            {'ohem_alpha': 0.5, 'ohem_gamma': 2.0, 'dice_weight': 0.6, 'focal_weight': 0.4},
            {'ohem_alpha': 0.25, 'ohem_gamma': 3.0, 'dice_weight': 0.6, 'focal_weight': 0.4},
            {'ohem_alpha': 0.25, 'ohem_gamma': 2.0, 'dice_weight': 0.8, 'focal_weight': 0.2},
        ]
        
        # PRIORITY 6: Data augmentation (generalization)
        augmentation_configs = [
            {'augmentation_level': 'basic'},
            {'augmentation_level': 'strong'},
            {'augmentation_level': 'none'},  # For comparison
        ]
        
        return {
            'learning_rates': learning_rates,
            'schedulers': schedulers,
            'optimizers': optimizers,
            'training_configs': training_configs,
            'loss_configs': loss_configs,
            'augmentation_configs': augmentation_configs
        }
    
    def generate_experiment_configs(self, search_space, max_experiments=50):
        """
        Generate experiment configurations using intelligent sampling.
        Balances thorough search with computational constraints.
        """
        configs = []
        
        # Full grid search for learning rates (highest priority)
        lr_combinations = list(itertools.product(
            search_space['learning_rates']['encoder_lr'],
            search_space['learning_rates']['decoder_lr']
        ))
        
        # Sample other hyperparameters to stay within budget
        import random
        random.seed(42)  # Reproducible sampling
        
        for encoder_lr, decoder_lr in lr_combinations:
            if len(configs) >= max_experiments:
                break
                
            # Sample one configuration from each category
            scheduler_config = random.choice(search_space['schedulers'])
            optimizer_config = random.choice(search_space['optimizers'])
            training_config = random.choice(search_space['training_configs'])
            loss_config = random.choice(search_space['loss_configs'])
            augmentation_config = random.choice(search_space['augmentation_configs'])
            
            # Create experiment configuration
            experiment_config = {
                'encoder_lr': encoder_lr,
                'decoder_lr': decoder_lr,
                **scheduler_config,
                **optimizer_config,
                **training_config,
                **loss_config,
                **augmentation_config,
                # Fixed parameters
                'batch_size': 4,  # Keep consistent with current setup
                'img_size': 512,
                'num_classes': 3,
                'use_ohem': True,
                'gradient_clip_norm': 1.0,  # Add gradient clipping for stability
                'seed': 42,  # Reproducible experiments
            }
            
            configs.append(experiment_config)
        
        # Add a few baseline configurations for comparison
        baseline_configs = [
            # Current best configuration (from 15.1% success) with warmup
            {
                'encoder_lr': 1e-4, 'decoder_lr': 2e-3, 'scheduler': 'cosine-warmup',
                'scheduler_factor': 1.0, 'warmup_steps': 100, 'optimizer': 'adamw', 'weight_decay': 1e-4,
                'beta1': 0.9, 'beta2': 0.999, 'epochs': 100, 'early_stopping_patience': 15,
                'ohem_alpha': 0.25, 'ohem_gamma': 2.0, 'dice_weight': 0.6, 'focal_weight': 0.4,
                'augmentation_level': 'basic', 'batch_size': 4, 'img_size': 512, 'num_classes': 3, 
                'use_ohem': True, 'gradient_clip_norm': 1.0, 'seed': 42, 'experiment_type': 'baseline_current_warmup'
            },
            # Original finetuning configuration (from run_finetuning.py)
            {
                'encoder_lr': 1e-5, 'decoder_lr': 5e-4, 'scheduler': 'plateau',
                'scheduler_factor': 0.5, 'scheduler_patience': 5, 'optimizer': 'adamw',
                'weight_decay': 1e-4, 'beta1': 0.9, 'beta2': 0.999, 'epochs': 100,
                'early_stopping_patience': 10, 'ohem_alpha': 0.25, 'ohem_gamma': 2.0,
                'dice_weight': 0.6, 'focal_weight': 0.4, 'augmentation_level': 'basic',
                'batch_size': 4, 'img_size': 512, 'num_classes': 3, 'use_ohem': True, 
                'gradient_clip_norm': None, 'seed': 42, 'experiment_type': 'baseline_original'
            }
        ]
        
        # Add baselines to the beginning
        configs = baseline_configs + configs[:max_experiments-len(baseline_configs)]
        
        logger.info(f"Generated {len(configs)} experiment configurations")
        return configs
    
    def run_single_experiment(self, config, experiment_id):
        """
        Run a single hyperparameter experiment.
        """
        experiment_name = f"exp_{experiment_id:03d}"
        if 'experiment_type' in config:
            experiment_name += f"_{config['experiment_type']}"
        
        logger.info(f"Starting experiment {experiment_name}")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        
        # Build command line arguments
        cmd = [
            sys.executable, "scripts/configurable_finetuning.py",
            "--experiment-name", experiment_name,
            "--encoder-lr", str(config['encoder_lr']),
            "--decoder-lr", str(config['decoder_lr']),
            "--scheduler", config['scheduler'],
            "--scheduler-factor", str(config['scheduler_factor']),
            "--optimizer", config['optimizer'],
            "--weight-decay", str(config['weight_decay']),
            "--beta1", str(config['beta1']),
            "--beta2", str(config['beta2']),
            "--epochs", str(config['epochs']),
            "--early-stopping-patience", str(config['early_stopping_patience']),
            "--ohem-alpha", str(config['ohem_alpha']),
            "--ohem-gamma", str(config['ohem_gamma']),
            "--dice-weight", str(config['dice_weight']),
            "--focal-weight", str(config['focal_weight']),
            "--batch-size", str(config['batch_size']),
            "--img-size", str(config['img_size']),
            "--num-classes", str(config['num_classes']),
            "--use-ohem",
        ]
        
        # Add optional parameters
        if 'scheduler_patience' in config:
            cmd.extend(["--scheduler-patience", str(config['scheduler_patience'])])
        if 'warmup_steps' in config:
            cmd.extend(["--warmup-steps", str(config['warmup_steps'])])
        if config.get('gradient_clip_norm'):
            cmd.extend(["--gradient-clip-norm", str(config['gradient_clip_norm'])])
        if 'augmentation_level' in config:
            cmd.extend(["--augmentation-level", config['augmentation_level']])
        if 'seed' in config:
            cmd.extend(["--seed", str(config['seed'])])
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            duration = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                # Try to extract best IoU from hyperparameter_config.json
                config_path = Path(f"work_dirs/configurable_finetuning/{experiment_name}/hyperparameter_config.json")
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        exp_config = json.load(f)
                        best_iou = exp_config.get('best_iou', 0.0)
                else:
                    best_iou = 0.0
                
                logger.info(f"Experiment {experiment_name} completed: {best_iou:.1%} IoU")
                
                return {
                    'experiment_id': experiment_id,
                    'experiment_name': experiment_name,
                    'config': config,
                    'best_iou': best_iou,
                    'duration_minutes': duration / 60,
                    'status': 'success',
                    'stdout': result.stdout[-1000:],  # Last 1000 chars
                    'stderr': result.stderr[-1000:] if result.stderr else ""
                }
            else:
                logger.error(f"Experiment {experiment_name} failed with return code {result.returncode}")
                return {
                    'experiment_id': experiment_id,
                    'experiment_name': experiment_name,
                    'config': config,
                    'best_iou': 0.0,
                    'duration_minutes': duration / 60,
                    'status': 'failed',
                    'stdout': result.stdout[-1000:],
                    'stderr': result.stderr[-1000:] if result.stderr else ""
                }
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Experiment {experiment_name} crashed: {e}")
            return {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'config': config,
                'best_iou': 0.0,
                'duration_minutes': duration / 60,
                'status': 'crashed',
                'error': str(e),
                'stdout': "",
                'stderr': ""
            }
    
    def run_sweep(self, max_experiments=30, max_parallel=2):
        """
        Run the complete hyperparameter sweep.
        """
        logger.info(f"Starting hyperparameter sweep with {max_experiments} experiments")
        logger.info(f"Parallel workers: {max_parallel}")
        logger.info(f"Results will be saved to: {self.sweep_dir}")
        
        # Generate experiment configurations
        search_space = self.define_search_space()
        configs = self.generate_experiment_configs(search_space, max_experiments)
        
        # Save sweep configuration
        sweep_config = {
            'sweep_id': self.sweep_id,
            'timestamp': datetime.now().isoformat(),
            'search_space': search_space,
            'total_experiments': len(configs),
            'max_parallel': max_parallel
        }
        
        with open(self.sweep_dir / "sweep_config.json", 'w') as f:
            json.dump(sweep_config, f, indent=2)
        
        # Run experiments
        start_time = time.time()
        
        if max_parallel == 1:
            # Sequential execution for debugging
            for i, config in enumerate(configs):
                result = self.run_single_experiment(config, i)
                self.results.append(result)
                self._save_intermediate_results()
        else:
            # Parallel execution for efficiency
            with ProcessPoolExecutor(max_workers=max_parallel) as executor:
                # Submit all experiments
                future_to_config = {
                    executor.submit(self.run_single_experiment, config, i): (config, i)
                    for i, config in enumerate(configs)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_config):
                    config, exp_id = future_to_config[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        self._save_intermediate_results()
                        
                        # Log progress
                        completed = len(self.results)
                        total = len(configs)
                        best_so_far = max(self.results, key=lambda x: x['best_iou'])
                        logger.info(f"Progress: [{completed}/{total}] - Best so far: {best_so_far['best_iou']:.1%} ({best_so_far['experiment_name']})")
                        
                    except Exception as e:
                        logger.error(f"Experiment {exp_id} generated exception: {e}")
        
        # Final analysis
        total_duration = time.time() - start_time
        self._analyze_results(total_duration)
        
        return self.results
    
    def _save_intermediate_results(self):
        """Save results after each experiment completion."""
        results_path = self.sweep_dir / "intermediate_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _analyze_results(self, total_duration):
        """
        Analyze sweep results and generate insights.
        """
        logger.info("=" * 80)
        logger.info("HYPERPARAMETER SWEEP ANALYSIS")
        logger.info("=" * 80)
        
        if not self.results:
            logger.error("No results to analyze!")
            return
        
        # Sort by performance
        successful_results = [r for r in self.results if r['status'] == 'success' and r['best_iou'] > 0]
        successful_results.sort(key=lambda x: x['best_iou'], reverse=True)
        
        logger.info(f"Total experiments: {len(self.results)}")
        logger.info(f"Successful experiments: {len(successful_results)}")
        logger.info(f"Failed experiments: {len(self.results) - len(successful_results)}")
        logger.info(f"Total duration: {total_duration/3600:.1f} hours")
        
        if successful_results:
            # Top 5 results
            logger.info("\n🏆 TOP 5 CONFIGURATIONS:")
            for i, result in enumerate(successful_results[:5]):
                config = result['config']
                logger.info(f"  {i+1}. {result['experiment_name']}: {result['best_iou']:.1%} IoU")
                logger.info(f"     LR: enc={config['encoder_lr']:.0e}, dec={config['decoder_lr']:.0e}")
                logger.info(f"     Scheduler: {config['scheduler']} (factor={config.get('scheduler_factor', 'N/A')})")
                logger.info(f"     Duration: {result['duration_minutes']:.0f} min")
            
            # Best configuration details
            best_result = successful_results[0]
            best_config = best_result['config']
            
            logger.info(f"\n🎯 BEST CONFIGURATION ({best_result['best_iou']:.1%} IoU):")
            for key, value in best_config.items():
                if key != 'experiment_type':
                    logger.info(f"  {key}: {value}")
            
            # Performance improvement analysis
            baseline_results = [r for r in successful_results if 'baseline' in r.get('config', {}).get('experiment_type', '')]
            if baseline_results:
                baseline_iou = max(r['best_iou'] for r in baseline_results)
                improvement = (best_result['best_iou'] - baseline_iou) / baseline_iou * 100
                logger.info(f"\n📈 IMPROVEMENT OVER BASELINE:")
                logger.info(f"  Baseline: {baseline_iou:.1%} IoU")
                logger.info(f"  Optimized: {best_result['best_iou']:.1%} IoU") 
                logger.info(f"  Improvement: +{improvement:.1f}%")
            
            # Hyperparameter insights
            self._extract_hyperparameter_insights(successful_results)
        
        # Save final results
        final_results = {
            'sweep_summary': {
                'sweep_id': self.sweep_id,
                'total_experiments': len(self.results),
                'successful_experiments': len(successful_results),
                'best_iou': successful_results[0]['best_iou'] if successful_results else 0.0,
                'total_duration_hours': total_duration / 3600,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': self.results,
            'top_5_configs': successful_results[:5] if successful_results else []
        }
        
        with open(self.sweep_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"\n📄 Results saved to: {self.sweep_dir}/final_results.json")
        logger.info("=" * 80)
    
    def _extract_hyperparameter_insights(self, results):
        """Extract insights about which hyperparameters matter most."""
        logger.info(f"\n🔍 HYPERPARAMETER INSIGHTS:")
        
        # Learning rate analysis
        lr_performance = {}
        for result in results:
            config = result['config']
            lr_key = f"enc_{config['encoder_lr']:.0e}_dec_{config['decoder_lr']:.0e}"
            if lr_key not in lr_performance:
                lr_performance[lr_key] = []
            lr_performance[lr_key].append(result['best_iou'])
        
        # Best learning rate combinations
        lr_avg_performance = {k: np.mean(v) for k, v in lr_performance.items()}
        best_lr = max(lr_avg_performance, key=lr_avg_performance.get)
        logger.info(f"  Best LR combination: {best_lr} (avg {lr_avg_performance[best_lr]:.1%} IoU)")
        
        # Scheduler analysis
        scheduler_performance = {}
        for result in results:
            scheduler = result['config']['scheduler']
            if scheduler not in scheduler_performance:
                scheduler_performance[scheduler] = []
            scheduler_performance[scheduler].append(result['best_iou'])
        
        for scheduler, ious in scheduler_performance.items():
            avg_iou = np.mean(ious)
            logger.info(f"  Scheduler '{scheduler}': avg {avg_iou:.1%} IoU ({len(ious)} experiments)")

def main():
    """Main function for running hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for ViT-Base optimization")
    
    parser.add_argument('--max-experiments', type=int, default=30,
                       help='Maximum number of experiments to run')
    parser.add_argument('--max-parallel', type=int, default=2,
                       help='Maximum parallel workers (be careful with GPU memory)')
    parser.add_argument('--save-dir', type=str, default='work_dirs/hyperparameter_sweep',
                       help='Base directory to save sweep results')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("🚀 HYPERPARAMETER SWEEP - PHASE 4A: VIT-BASE OPTIMIZATION")
    print("=" * 100)
    print(f"Max experiments: {args.max_experiments}")
    print(f"Parallel workers: {args.max_parallel}")
    print(f"Expected duration: {args.max_experiments * 1.5 / args.max_parallel:.1f} hours")
    print("=" * 100)
    
    # Create and run sweep
    sweep = HyperparameterSweep(base_save_dir=args.save_dir)
    results = sweep.run_sweep(
        max_experiments=args.max_experiments,
        max_parallel=args.max_parallel
    )
    
    # Final summary
    if results:
        successful_results = [r for r in results if r['status'] == 'success' and r['best_iou'] > 0]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['best_iou'])
            
            print(f"\n🎉 SWEEP COMPLETED SUCCESSFULLY!")
            print(f"🏆 Best result: {best_result['best_iou']:.1%} IoU ({best_result['experiment_name']})")
            print(f"📊 Success rate: {len(successful_results)}/{len(results)} experiments")
            print(f"🎯 Ready for Phase 4B: Error analysis on optimized model")
        else:
            print(f"\n❌ No successful experiments. Check logs for issues.")
    else:
        print(f"\n❌ Sweep failed to produce results. Check configuration.")

if __name__ == "__main__":
    # Add numpy import for analysis
    import numpy as np
    main()