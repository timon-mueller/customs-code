import os
import yaml
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class ModelSaver:
    def __init__(self, project_root, config, idx_to_chapter):
        run_name = config.get('training', {}).get('run_name', config['model']['name'])
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        folder_name = f"{run_name}_{timestamp}"
        self.save_dir = os.path.join(project_root, "models", "checkpoints", folder_name)
        
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Model outputs & logs will be saved to: {self.save_dir}")

        # --- TENSORBOARD SETUP ---
        # Initializes the logger directly inside your specific run's folder
        self.writer = SummaryWriter(log_dir=self.save_dir)

        mapping_path = os.path.join(self.save_dir, "class_mapping.json")
        with open(mapping_path, "w", encoding='utf-8') as f:
            import json
            json.dump(idx_to_chapter, f, indent=4)
            
        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(config_path, "w", encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr=None):
        """
        Records the epoch metrics to TensorBoard.
        """
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        self.writer.add_scalar('Hyperparameters/Learning_Rate', lr, epoch)

    def save_epoch(self, model, epoch):
        weights_filename = f"model_weights_epoch_{epoch}.pt"
        weights_path = os.path.join(self.save_dir, weights_filename)
        torch.save(model.state_dict(), weights_path)
        print(f"  -> Saved weights for Epoch {epoch} to {weights_filename}")

    def close(self):
        """Safely closes the TensorBoard writer."""
        self.writer.close()