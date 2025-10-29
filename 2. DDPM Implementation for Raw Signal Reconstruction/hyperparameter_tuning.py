"""
Hyperparameter tuning classes and functions with configurable epochs
"""
import os
import logging
import time
import json
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import get_safe_parameter_bounds, get_gpu_memory_info, estimate_model_memory, clear_gpu_memory
from dataset import ECoGDBSDataset
from trainer import train_ecog_dbs_model

class HyperparameterTuner:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        base_config,
        n_trials=10,
        timeout=3600*4,
        output_dir=None,
        tuning_epochs=30,
        final_epochs=100
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.base_config = base_config
        self.n_trials = n_trials
        self.timeout = timeout
        self.output_dir = output_dir
        self.tuning_epochs = tuning_epochs
        self.final_epochs = final_epochs
        self.trial_results = []
        self.best_trial = None
        self.study = None
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging to both file and console"""
        if self.output_dir:
            log_file = os.path.join(self.output_dir, 'hyperparameter_tuning.log')
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s'
            )
    
    def log_trial_start(self, trial_number, parameters):
        """Log trial start with parameters"""
        logging.info(f"\n{'='*80}")
        logging.info(f"Starting Trial {trial_number}")
        logging.info("Parameters:")
        for param, value in parameters.items():
            logging.info(f"  {param}: {value}")
        logging.info(f"{'='*80}\n")
    
    def log_trial_result(self, trial_number, parameters, value, duration):
        """Log trial results"""
        logging.info(f"\n{'-'*80}")
        logging.info(f"Trial {trial_number} completed")
        logging.info(f"Validation Loss: {value:.6f}")
        logging.info(f"Duration: {duration:.2f} seconds")
        logging.info(f"{'-'*80}\n")
        
        self.trial_results.append({
            'trial_number': trial_number,
            'validation_loss': value,
            'duration': duration,
            **parameters
        })
    
    def save_results(self):
        """Save tuning results to files"""
        if not self.output_dir:
            return
            
        # Save trial results as CSV
        df = pd.DataFrame(self.trial_results)
        csv_path = os.path.join(self.output_dir, 'trial_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Save study statistics
        stats = {
            'best_params': self.best_trial.params,
            'best_value': self.best_trial.value,
            'n_trials': len(self.study.trials),
            'study_duration': sum(result['duration'] for result in self.trial_results),
            'tuning_epochs': self.tuning_epochs,
            'final_epochs': self.final_epochs
        }
        
        stats_path = os.path.join(self.output_dir, 'study_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        logging.info(f"\nResults saved to {self.output_dir}")

class MemorySafeHyperparameterTuner(HyperparameterTuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get available GPU memory
        if torch.cuda.is_available():
            dedicated, total = get_gpu_memory_info()
            self.available_gpu_memory = total - dedicated
        else:
            self.available_gpu_memory = 0
        
        # Get parameter bounds based on available memory
        self.param_bounds = get_safe_parameter_bounds(self.available_gpu_memory)
        
        logging.info(f"\nInitializing Memory-Safe Tuner:")
        logging.info(f"Available GPU Memory: {self.available_gpu_memory:.2f} GB")
        logging.info(f"Using parameter bounds: {self.param_bounds}")
        logging.info(f"Tuning epochs: {self.tuning_epochs}")
        logging.info(f"Final epochs: {self.final_epochs}")
    
    def check_memory_usage(self):
        """Check memory usage with adjusted thresholds"""
        dedicated, total = get_gpu_memory_info()
        
        if dedicated > 16 or total > 20:  # Conservative limits
            logging.warning(f"Stopping trial - Memory usage too high:")
            logging.warning(f"Dedicated: {dedicated:.2f} GB")
            logging.warning(f"Total: {total:.2f} GB")
            return True
        return False
    
    def objective(self, trial):
        """Memory-safe objective function"""
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Suggest parameters within safe bounds
        parameters = {
            'hidden_channel': trial.suggest_int('hidden_channel', 
                                              self.param_bounds['hidden_channel'][0],
                                              self.param_bounds['hidden_channel'][1]),
            'num_blocks': trial.suggest_int('num_blocks',
                                          self.param_bounds['num_blocks'][0],
                                          self.param_bounds['num_blocks'][1]),
            'diffusion_steps': trial.suggest_int('diffusion_steps',
                                               self.param_bounds['diffusion_steps'][0],
                                               self.param_bounds['diffusion_steps'][1]),
            'batch_size': trial.suggest_int('batch_size',
                                          self.param_bounds['batch_size'][0],
                                          self.param_bounds['batch_size'][1]),
            'start_beta': trial.suggest_float('start_beta', 1e-5, 1e-3, log=True),
            'end_beta': trial.suggest_float('end_beta', 1e-2, 5e-2),
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        }
        
        # Check initial memory estimate
        estimated_memory = estimate_model_memory(
            parameters['hidden_channel'],
            parameters['num_blocks'],
            parameters['batch_size'],
            self.base_config.dataset.signal_length
        )
        
        logging.info(f"\nMemory Analysis:")
        logging.info(f"Estimated memory requirement: {estimated_memory:.2f} GB")
        logging.info(f"Current GPU memory usage: {get_gpu_memory_info()[0]:.2f} GB")
        
        if estimated_memory > 20:
            logging.warning("Skipping trial - estimated memory requirement too high")
            raise optuna.TrialPruned()
        
        self.log_trial_start(trial.number, parameters)
        
        try:
            # Update config with suggested parameters
            config = self.base_config.copy()
            config.optimizer.num_epochs = self.tuning_epochs  # Use tuning epochs
            
            for param, value in parameters.items():
                if param in ['hidden_channel', 'num_blocks']:
                    config.network[param] = value
                elif param in ['diffusion_steps', 'start_beta', 'end_beta']:
                    config.diffusion[param] = value
                elif param == 'lr':
                    config.optimizer.lr = value
                elif param == 'batch_size':
                    config.optimizer.train_batch_size = value
            
            # Train model
            diffusion, _, _ = train_ecog_dbs_model(
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                config=config
            )
            
            # Evaluate on validation set
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=parameters['batch_size'],
                shuffle=False
            )
            
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if self.check_memory_usage():
                        raise optuna.TrialPruned()
                    
                    sig_batch = batch["signal"].to(self.device)
                    cond_batch = batch["cond"].to(self.device)
                    loss = diffusion.train_batch(sig_batch, cond=cond_batch)
                    val_losses.append(torch.mean(loss).item())
            
            val_loss = sum(val_losses) / len(val_losses)
            
            duration = time.time() - start_time
            self.log_trial_result(trial.number, parameters, val_loss, duration)
            
            # Clean up
            del diffusion
            clear_gpu_memory()
            
            return val_loss
            
        except Exception as e:
            logging.error(f"Error in trial: {str(e)}")
            clear_gpu_memory()
            raise optuna.TrialPruned()
    
    def tune(self):
        """Run hyperparameter tuning"""
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        self.study = study
        self.best_trial = study.best_trial
        
        self.save_results()
        
        return study.best_params

def tune_hyperparameters_safely(
    ecog_data, 
    dbs_data, 
    base_config, 
    output_dir=None, 
    n_trials=10, 
    timeout=3600*4,
    tuning_epochs=30,
    final_epochs=100
):
    """Main function for memory-safe hyperparameter tuning"""
    # Create dataset
    dataset = ECoGDBSDataset(ecog_data, dbs_data)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Initialize tuner
    tuner = MemorySafeHyperparameterTuner(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        base_config=base_config,
        n_trials=n_trials,
        timeout=timeout,
        output_dir=output_dir,
        tuning_epochs=tuning_epochs,
        final_epochs=final_epochs
    )
    
    # Run tuning
    best_params = tuner.tune()
    
    # Update config with best parameters and final_epochs for final training
    final_config = base_config.copy()
    final_config.optimizer.num_epochs = final_epochs
    
    for param, value in best_params.items():
        if param in ['hidden_channel', 'num_blocks']:
            final_config.network[param] = value
        elif param in ['diffusion_steps', 'start_beta', 'end_beta']:
            final_config.diffusion[param] = value
        elif param == 'lr':
            final_config.optimizer.lr = value
        elif param == 'batch_size':
            final_config.optimizer.train_batch_size = value
    
    return best_params, val_dataset, final_config