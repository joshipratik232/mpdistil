"""Main API class for MPDistil.

This module provides the user-facing MPDistil class with a simple
scikit-learn-style API for training student models.
"""

from typing import Optional, Dict, List
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from .models import FineTunedModel, ActionPredictor
from .trainer import MPDistilTrainer
from .config import TrainingConfig, TaskConfig
from .data_utils import prepare_task_loaders, validate_dataloader


class MPDistil:
    """Meta-Policy Knowledge Distillation.
    
    A teacher-student collaborative knowledge distillation framework with
    curriculum learning for training compact student models that can outperform
    larger teacher models.
    
    The training process consists of 4 phases:
    1. Teacher fine-tuning on the main task
    2. Student knowledge distillation with Patient KD
    3. Meta-teacher learning (collaborative or competitive)
    4. Curriculum learning with reinforcement learning
    
    Example:
        >>> from mpdistil import MPDistil
        >>> 
        >>> model = MPDistil(
        ...     task_name='CB',
        ...     num_labels=3,
        ...     teacher_model='bert-base-uncased',
        ...     student_model='bert-base-uncased',
        ...     student_layers=6
        ... )
        >>> 
        >>> history = model.fit(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     meta_loaders={'RTE': rte_loader},
        ...     teacher_epochs=10,
        ...     student_epochs=10,
        ...     num_episodes=200
        ... )
        >>> 
        >>> model.save_student('./my_student_model')
    
    Args:
        task_name: Name of the main task
        num_labels: Number of output classes
        teacher_model: HuggingFace model name for teacher (default: 'bert-base-uncased')
        student_model: HuggingFace model name for student (default: 'bert-base-uncased')
        student_layers: Number of layers for student encoder (default: -1, -1 means use original architecture)
        device: Device to use ('auto', 'cuda', 'cpu')
        output_dir: Directory for checkpoints and outputs
    """
    
    def __init__(
        self,
        task_name: str,
        num_labels: Optional[int] = None,
        task_type: str = 'classification',
        teacher_model: str = 'bert-base-uncased',
        student_model: str = 'bert-base-uncased',
        student_layers: int = -1,
        device: str = 'auto',
        output_dir: str = './mpdistil_outputs'
    ):
        self.task_name = task_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.output_dir = output_dir
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model)
        # Ensure tokenizer has pad token (needed for GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store model configs
        self._teacher_model_name = teacher_model
        self._student_model_name = student_model
        self._student_layers = student_layers
        
        # Models will be initialized in fit() after we know all task names
        self.teacher_model = None
        self.student_model = None
        self.action_model = None
        
        # Store task info (will be updated in fit())
        self.task_names = [task_name]
        # For language modeling, num_labels might be None
        if task_type == 'classification':
            if num_labels is None:
                raise ValueError("num_labels must be provided for classification tasks")
            self.label_nums = {task_name.lower(): num_labels}
        else:
            # Language modeling doesn't use label_nums
            self.label_nums = {task_name.lower(): 0}  # Placeholder
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _initialize_models(self):
        """Initialize teacher, student, and action models."""
        # Initialize teacher config
        teacher_config = AutoConfig.from_pretrained(
            self._teacher_model_name
        )
        if self.task_type == 'classification' and self.num_labels:
            teacher_config.num_labels = self.num_labels
        
        # Initialize student config
        student_config = AutoConfig.from_pretrained(
            self._student_model_name
        )
        if self.task_type == 'classification' and self.num_labels:
            student_config.num_labels = self.num_labels
        
        # Only slice student layers if explicitly requested (student_layers > 0)
        if self._student_layers > 0 and hasattr(student_config, 'num_hidden_layers'):
            print(f"Slicing student model to {self._student_layers} layers (original: {student_config.num_hidden_layers})")
            student_config.num_hidden_layers = self._student_layers
        elif self._student_layers == -1:
            print(f"Using original student architecture ({student_config.num_hidden_layers} layers, no slicing)")
        
        # Initialize teacher model
        self.teacher_model = FineTunedModel(
            self.task_names,
            self.label_nums,
            teacher_config,
            pretrained_model_name=self._teacher_model_name,
            task_type=self.task_type
        ).to(self.device)
        
        # Initialize student model
        self.student_model = FineTunedModel(
            self.task_names,
            self.label_nums,
            student_config,
            pretrained_model_name=self._student_model_name,
            task_type=self.task_type
        ).to(self.device)
        
        # Initialize action model
        self.action_model = ActionPredictor(
            d_model=768,  # BERT hidden size
            num_actions=len(self.task_names)
        ).to(self.device)
        
        # Print model sizes
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
        student_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        action_params = sum(p.numel() for p in self.action_model.parameters() if p.requires_grad)
        
        print(f"\nModel Sizes:")
        print(f"  Teacher: {teacher_params:,} parameters")
        print(f"  Student: {student_params:,} parameters ({student_params/teacher_params*100:.1f}% of teacher)")
        print(f"  Action:  {action_params:,} parameters")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        meta_loaders: Optional[Dict[str, DataLoader]] = None,
        config: Optional[TrainingConfig] = None,
        report_to: Optional[str] = None,
        **config_kwargs
    ) -> Dict:
        """Train the MPDistil model.
        
        This method runs all 4 phases of training:
        1. Teacher fine-tuning
        2. Student knowledge distillation with PKD
        3. Meta-teacher learning
        4. Curriculum learning (if meta_loaders provided)
        
        Args:
            train_loader: Training DataLoader
                Expected batch format: (input_ids, attention_mask, token_type_ids, labels)
            val_loader: Validation DataLoader
            test_loader: Optional test DataLoader (no labels expected)
            meta_loaders: Optional dict of auxiliary task DataLoaders for curriculum learning
                Format: {'TaskName': dataloader, ...}
            config: TrainingConfig object (creates default if None)
            report_to: Where to log metrics ('wandb', 'tensorboard', 'none', or None)
            **config_kwargs: Override specific config parameters
                Examples: teacher_epochs=5, student_lr=1e-4, alpha=0.7, report_to='wandb'
        
        Returns:
            Dictionary with training history from all phases
            Keys: 'phase1', 'phase2', 'phase3', 'phase4' (if applicable)
        
        Example:
            >>> history = model.fit(
            ...     train_loader=train_dl,
            ...     val_loader=val_dl,
            ...     meta_loaders={'RTE': rte_dl, 'BoolQ': boolq_dl},
            ...     teacher_epochs=5,
            ...     student_epochs=5,
            ...     alpha=0.7,
            ...     beta=50.0,
            ...     num_episodes=100
            ... )
        """
        # Validate dataloaders
        print("\nValidating DataLoaders...")
        validate_dataloader(train_loader, split='train', check_labels=True)
        validate_dataloader(val_loader, split='val', check_labels=True)
        if test_loader is not None:
            validate_dataloader(test_loader, split='test', check_labels=False)
        
        # Create or update config
        if config is None:
            config = TrainingConfig()
        if config_kwargs:
            config.merge(**config_kwargs)
        
        # Override report_to if provided as parameter
        if report_to is not None:
            config.report_to = report_to
        
        # Initialize WandB if requested
        if config.report_to == 'wandb':
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.wandb_run_name or f"{self.task_name}_{self.task_type}",
                    config=config.to_dict()
                )
                print("✓ WandB logging enabled")
            except ImportError:
                print("⚠ WandB not installed. Install with: pip install wandb")
                config.report_to = None
        
        # Set seed
        self._set_seed(config.seed)
        
        # Update output dir
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Prepare task_loaders internal structure
        print("\nPreparing task loaders...")
        task_loaders, label_nums = prepare_task_loaders(
            main_task_name=self.task_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            meta_loaders=meta_loaders,
            num_labels=self.num_labels,
            auto_split_held=True,
            held_ratio=0.2,
            seed=config.seed
        )
        
        # Update task names and label nums
        self.task_names = list(task_loaders.keys())
        self.label_nums = label_nums
        
        print(f"Tasks: {self.task_names}")
        print(f"Label counts: {self.label_nums}")
        
        # Initialize models NOW (after we know all task names)
        print("\nInitializing models...")
        self._initialize_models()
        
        # Create task config
        task_config = TaskConfig(
            task_name=self.task_name,
            num_labels=self.num_labels if self.task_type == 'classification' else 0,
            task_type=self.task_type
        )
        
        # Initialize trainer
        trainer = MPDistilTrainer(
            teacher=self.teacher_model,
            student=self.student_model,
            action_model=self.action_model,
            config=config,
            task_config=task_config,
            device=self.device
        )
        
        # Run all phases
        print("\n" + "="*60)
        print("Starting MPDistil Training")
        print("="*60)
        
        history = trainer.run_all_phases(task_loaders, label_nums)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        return history
    
    def save_student(self, save_directory: str):
        """Save student model in HuggingFace format.
        
        The saved model can be loaded with HuggingFace's `from_pretrained()`.
        
        Args:
            save_directory: Directory to save model
        
        Example:
            >>> model.save_student('./my_student_model')
            >>> # Later, load with:
            >>> # from transformers import AutoModel
            >>> # student = AutoModel.from_pretrained('./my_student_model')
        """
        if self.student_model is None:
            raise RuntimeError("No trained student model to save. Call fit() first.")
        
        os.makedirs(save_directory, exist_ok=True)
        
        print(f"\nSaving student model to {save_directory}...")
        self.student_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print("Model saved successfully!")
    
    def load_student(self, load_directory: str):
        """Load a saved student model.
        
        Args:
            load_directory: Directory containing saved model
        
        Example:
            >>> model.load_student('./my_student_model')
        """
        print(f"\nLoading student model from {load_directory}...")
        
        self.student_model = FineTunedModel.from_pretrained(
            load_directory,
            self.task_names,
            self.label_nums
        )
        self.student_model.to(self.device)
        
        print("Model loaded successfully!")
    
    def predict(self, test_loader: DataLoader) -> List[int]:
        """Make predictions on test data.
        
        Args:
            test_loader: Test DataLoader (without labels)
        
        Returns:
            List of predicted class indices
        
        Example:
            >>> predictions = model.predict(test_loader)
            >>> print(predictions[:10])  # First 10 predictions
        """
        if self.student_model is None:
            raise RuntimeError("No trained student model. Call fit() first.")
        
        self.student_model.eval()
        all_preds = []
        
        print("\nGenerating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
                
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                
                outputs = self.student_model(
                    src=input_ids,
                    task_name=self.task_name.lower(),
                    mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=False
                )
                
                preds = outputs[0].argmax(-1)
                all_preds.extend(preds.cpu().numpy().tolist())
        
        print(f"Generated {len(all_preds)} predictions")
        return all_preds
    
    def save_predictions(
        self,
        predictions: List[int],
        save_path: str,
        label_mapping: Optional[Dict[int, str]] = None
    ):
        """Save predictions to TSV file.
        
        Args:
            predictions: List of predictions
            save_path: Path to save TSV file
            label_mapping: Optional mapping from indices to string labels
                Example: {0: 'entailment', 1: 'not_entailment'}
        
        Example:
            >>> preds = model.predict(test_loader)
            >>> model.save_predictions(
            ...     preds,
            ...     'predictions.tsv',
            ...     label_mapping={0: 'entailment', 1: 'contradiction', 2: 'neutral'}
            ... )
        """
        import pandas as pd
        
        df = pd.DataFrame({
            'index': range(len(predictions)),
            'prediction': predictions
        })
        
        if label_mapping is not None:
            df['prediction'] = df['prediction'].map(label_mapping)
        
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        df.to_csv(save_path, sep='\t', index=False)
        
        print(f"Predictions saved to {save_path}")
    
    @property
    def student(self):
        """Access to student model for advanced usage."""
        return self.student_model
    
    @property
    def teacher(self):
        """Access to teacher model for advanced usage."""
        return self.teacher_model
    
    @property
    def action(self):
        """Access to action model for advanced usage."""
        return self.action_model
