"""Configuration dataclasses for MPDistil.

This module provides typed configuration objects for training
and task-specific settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict


@dataclass
class TrainingConfig:
    """Training hyperparameters for all phases.
    
    Attributes:
        teacher_epochs: Number of epochs for teacher fine-tuning (Phase 1)
        teacher_lr: Learning rate for teacher
        skip_teacher_training: Skip Phase 1 if teacher already trained
        
        student_epochs: Number of epochs for student PKD (Phase 2)
        student_lr: Learning rate for student
        alpha: Weight for soft loss (0-1, remaining weight for hard loss)
        beta: Weight for PKD (patient knowledge distillation) loss
        temperature: Temperature for soft target generation
        
        meta_lr: Learning rate for meta-teacher (Phase 3)
        use_competitive_loss: Use competitive loss instead of collaborative
        
        num_episodes: Number of curriculum learning episodes (Phase 4)
        reward_type: Type of reward ('binary' or 'real')
        gamma: Discount factor for reward computation
        
        batch_size: Batch size for all phases
        max_grad_norm: Gradient clipping threshold
        weight_decay: AdamW weight decay
        warmup_steps: Linear warmup steps (currently unused)
        seed: Random seed for reproducibility
        
        device: Device to use ('auto', 'cuda', 'cpu')
        fp16: Use mixed precision training (requires APEX)
        fp16_opt_level: APEX optimization level
        
        wandb_logging: Enable Weights & Biases logging
        wandb_project: W&B project name
        verbose: Show progress bars and detailed logging
        
        output_dir: Directory for checkpoints and outputs
        save_checkpoints: Save checkpoints during training
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    
    # Phase 1: Teacher Fine-tuning
    teacher_epochs: int = 10
    teacher_lr: float = 2e-5
    skip_teacher_training: bool = False
    
    # Phase 2: Student PKD
    student_epochs: int = 10
    student_lr: float = 3e-5
    alpha: float = 0.5
    beta: float = 100.0
    temperature: float = 5.0
    
    # Phase 3: Meta-Teacher
    meta_lr: float = 1e-3
    use_competitive_loss: bool = False
    
    # Phase 4: Curriculum Learning
    num_episodes: int = 200
    reward_type: Literal['binary', 'real'] = 'binary'
    gamma: float = 0.99
    
    # General Training
    batch_size: int = 8
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 0
    seed: int = 42
    
    # System
    device: str = 'auto'
    fp16: bool = False
    fp16_opt_level: str = 'O1'
    
    # Logging
    wandb_logging: bool = False
    wandb_project: str = 'mpdistil'
    verbose: bool = True
    
    # Checkpointing
    output_dir: str = './mpdistil_outputs'
    save_checkpoints: bool = True
    resume_from_checkpoint: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary.
        
        Returns:
            Dictionary of all configuration parameters
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters
            
        Returns:
            TrainingConfig instance
        """
        return cls(**config_dict)
    
    def merge(self, **kwargs):
        """Update specific parameters.
        
        Args:
            **kwargs: Parameters to override
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")


@dataclass
class TaskConfig:
    """Task-specific configuration.
    
    Attributes:
        task_name: Name of the task
        num_labels: Number of output classes
        output_mode: 'classification' or 'regression'
        metric: Primary metric to optimize ('accuracy', 'f1', 'mcc', 'correlation')
        label_mapping: Optional mapping from class indices to string labels
    """
    
    task_name: str
    num_labels: int
    output_mode: Literal['classification', 'regression'] = 'classification'
    metric: str = 'accuracy'
    label_mapping: Optional[Dict[int, str]] = None
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary.
        
        Returns:
            Dictionary of all configuration parameters
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TaskConfig':
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters
            
        Returns:
            TaskConfig instance
        """
        return cls(**config_dict)
    
    @property
    def is_regression(self) -> bool:
        """Check if this is a regression task.
        
        Returns:
            True if regression, False if classification
        """
        return self.output_mode == 'regression'
