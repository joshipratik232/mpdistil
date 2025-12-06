"""Model architectures for MPDistil.

This module contains the core model classes:
- SequenceClassificationHead: Task-specific output heads
- ActionPredictor: Policy network for curriculum learning
- FineTunedModel: Main teacher/student model wrapper
"""

from typing import Optional, Tuple, Dict
import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from copy import deepcopy as cp


class SequenceClassificationHead(nn.Module):
    """Classification head for sequence classification tasks.
    
    Args:
        hidden_size: Size of hidden layer
        num_labels: Number of output classes
        dropout_p: Dropout probability (default: 0.1)
    """
    
    def __init__(self, hidden_size: int, num_labels: int, dropout_p: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_weights()

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pooled_output: Pooled output from encoder [batch_size, hidden_size]
            
        Returns:
            Logits of shape [batch_size, num_labels]
        """
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def _init_weights(self):
        """Initialize weights with small random values."""
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class ActionPredictor(nn.Module):
    """Policy network for curriculum learning task selection.
    
    Predicts which auxiliary task to sample from during meta-learning.
    
    Args:
        d_model: Input dimension (default: 768 for BERT)
        num_actions: Number of tasks to choose from
    """
    
    def __init__(self, d_model: int = 768, num_actions: int = 8):
        super(ActionPredictor, self).__init__()
        self.action_predictor = nn.Linear(d_model, num_actions)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            state_tensor: Model state [batch_size, d_model]
            
        Returns:
            Action probabilities [batch_size, num_actions]
        """
        actions = torch.nn.Softmax(-1)(self.action_predictor(state_tensor))
        return actions


class FineTunedModel(nn.Module):
    """Multi-task model with task-specific heads.
    
    Wraps a pretrained encoder (e.g., BERT) with multiple task-specific
    classification heads. Used for both teacher and student models.
    
    Args:
        tasks: List of task names
        label_nums: Dict mapping task names (lowercase) to number of labels
        config: HuggingFace model config
        pretrained_model_name: Name of pretrained model to load
        tf_checkpoint: Optional TensorFlow checkpoint path (deprecated)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        tasks: list,
        label_nums: Dict[str, int],
        config: AutoConfig,
        pretrained_model_name: str = 'bert-base-uncased',
        tf_checkpoint: Optional[str] = None,
        dropout: float = 0.1
    ):
        super(FineTunedModel, self).__init__()

        self.config = config
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.drop = nn.Dropout(dropout)
        
        # Create task-specific output heads
        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = SequenceClassificationHead(
                self.encoder.config.hidden_size,
                label_nums[task.lower()]
            )
            self.output_heads[task.lower()] = decoder
        
        # State predictor for meta-learning
        self.state_predictor = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        task_name: str,
        src: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        pooled_output: Optional[torch.Tensor] = None,
        discriminator: bool = False,
        output_hidden_states: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.
        
        Args:
            task_name: Name of task (lowercase)
            src: Input IDs [batch_size, seq_length]
            mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length]
            pooled_output: Pre-computed pooled output (for discriminator mode)
            discriminator: If True, use pooled_output directly
            output_hidden_states: If True, return all hidden states
            
        Returns:
            Tuple of (logits, features, model_state, pooled_output):
            - logits: Task predictions [batch_size, num_labels]
            - features: Hidden states from all layers (if output_hidden_states=True)
            - model_state: Model state for action predictor
            - pooled_output: Pooled encoder output
        """
        if discriminator == False:
            outputs = self.encoder(
                src,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                output_hidden_states=output_hidden_states
            )
            
            encoder_output = outputs[0]
            pooled_output = outputs[1]
        
        # Get task-specific predictions
        out = self.output_heads[task_name.lower()](pooled_output)
        
        # Apply ReLU for regression tasks
        if task_name == 'sts-b':
            out = nn.ReLU()(out)
        
        # Compute model state for meta-learning
        model_state = self.state_predictor(
            self.encoder.pooler.dense.weight
        ).reshape(1, -1)
        
        features = None
        if output_hidden_states == True and discriminator == False:
            # Extract features from intermediate layers (excluding first and last)
            features = torch.cat(
                outputs[-1][1:-1], dim=0
            ).view(
                self.config.num_hidden_layers - 1,
                -1,
                src.size()[1],
                self.config.hidden_size
            )[:, :, 0]
        
        if discriminator == False:
            return (out, features, model_state, pooled_output)
        else:
            return (out, pooled_output, model_state, pooled_output)
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory.
        
        Args:
            save_directory: Path to save directory
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save encoder
        self.encoder.save_pretrained(save_directory)
        
        # Save full model state dict
        torch.save(
            self.state_dict(),
            os.path.join(save_directory, 'pytorch_model.bin')
        )
        
        # Save config
        self.config.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, tasks: list, label_nums: Dict[str, int]):
        """Load model from directory.
        
        Args:
            load_directory: Path to load directory
            tasks: List of task names
            label_nums: Dict mapping task names to label counts
            
        Returns:
            Loaded FineTunedModel instance
        """
        import os
        
        # Load config
        config = AutoConfig.from_pretrained(load_directory)
        
        # Create model
        model = cls(tasks, label_nums, config)
        
        # Load state dict
        state_dict = torch.load(
            os.path.join(load_directory, 'pytorch_model.bin'),
            map_location='cpu'
        )
        model.load_state_dict(state_dict)
        
        return model
