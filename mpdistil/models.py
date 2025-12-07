"""Model architectures for MPDistil.

This module contains the core model classes:
- SequenceClassificationHead: Task-specific output heads for classification
- ActionPredictor: Policy network for curriculum learning
- FineTunedModel: Main teacher/student model wrapper (supports both encoder and decoder models)
"""

from typing import Optional, Tuple, Dict, Literal
import torch
from torch import nn
from transformers import (
    AutoModel, AutoConfig, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
import numpy as np


def detect_model_type(config: AutoConfig) -> Literal['encoder', 'decoder', 'encoder-decoder']:
    """Detect if model is encoder-only, decoder-only, or encoder-decoder.
    
    Args:
        config: HuggingFace model config
        
    Returns:
        'encoder', 'decoder', or 'encoder-decoder'
    """
    # Check for encoder-decoder
    if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
        return 'encoder-decoder'
    
    # Check for common decoder-only models
    decoder_types = ['gpt', 'gpt2', 'gpt_neo', 'gpt_neox', 'llama', 'mistral', 'falcon', 'opt']
    if any(dt in config.model_type.lower() for dt in decoder_types):
        return 'decoder'
    
    # Default to encoder (BERT, RoBERTa, etc.)
    return 'encoder'


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
    
    Supports both encoder models (BERT, RoBERTa) and decoder models (GPT-2, Llama).
    Can handle classification and language modeling tasks.
    
    Args:
        tasks: List of task names
        label_nums: Dict mapping task names (lowercase) to number of labels (for classification)
        config: HuggingFace model config
        pretrained_model_name: Name of pretrained model to load
        task_type: 'classification' or 'language_modeling'
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        tasks: list,
        label_nums: Dict[str, int],
        config: AutoConfig,
        pretrained_model_name: str = 'bert-base-uncased',
        task_type: str = 'classification',
        dropout: float = 0.1
    ):
        super(FineTunedModel, self).__init__()

        self.config = config
        self.task_type = task_type
        self.model_type = detect_model_type(config)
        
        # Load appropriate model based on task type
        if task_type == 'language_modeling':
            # For LM, always use CausalLM
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, config=config)
            self.output_heads = None  # Use model's built-in LM head
        else:
            # For classification
            if self.model_type == 'encoder':
                # Encoder models: use AutoModel + custom heads
                self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
                # Create task-specific output heads
                self.output_heads = nn.ModuleDict()
                for task in tasks:
                    decoder = SequenceClassificationHead(
                        self.model.config.hidden_size,
                        label_nums[task.lower()],
                        dropout_p=dropout
                    )
                    self.output_heads[task.lower()] = decoder
            else:
                # Decoder models for classification: add classification head
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, config=config)
                self.output_heads = nn.ModuleDict()
                for task in tasks:
                    decoder = SequenceClassificationHead(
                        self.model.config.hidden_size,
                        label_nums[task.lower()],
                        dropout_p=dropout
                    )
                    self.output_heads[task.lower()] = decoder
        
        self.drop = nn.Dropout(dropout)
        
        # State predictor for meta-learning (use first layer weights as proxy)
        self.state_predictor = nn.Linear(config.hidden_size, 1)
    
    def _get_pooled_output(self, outputs, attention_mask=None):
        """Get pooled output from model outputs.
        
        For encoder models: use pooler_output or first token
        For decoder models: use last token or mean pooling
        """
        if self.model_type == 'encoder':
            # Try to get pooler_output (BERT-style)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                # Fallback: use [CLS] token (first token)
                return outputs.last_hidden_state[:, 0, :]
        else:
            # Decoder model: use last non-padding token
            if attention_mask is not None:
                # Get last non-padding token for each sequence
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = outputs.last_hidden_state.shape[0]
                return outputs.last_hidden_state[
                    torch.arange(batch_size, device=outputs.last_hidden_state.device), 
                    sequence_lengths
                ]
            else:
                # No mask: use last token
                return outputs.last_hidden_state[:, -1, :]
    
    def _get_model_state(self):
        """Get model state for action predictor.
        
        Uses first layer weights as a representation of model state.
        """
        # Get first parameter (works for any model)
        first_param = next(self.model.parameters())
        # Take mean across dimensions to get a hidden_size vector
        if first_param.dim() > 1:
            state_vector = first_param.mean(dim=0)
            if state_vector.dim() > 1:
                state_vector = state_vector.mean(dim=0)
        else:
            state_vector = first_param
        
        # Ensure it's the right size
        if state_vector.shape[0] != self.config.hidden_size:
            # Pad or truncate to hidden_size
            if state_vector.shape[0] < self.config.hidden_size:
                state_vector = torch.nn.functional.pad(
                    state_vector, 
                    (0, self.config.hidden_size - state_vector.shape[0])
                )
            else:
                state_vector = state_vector[:self.config.hidden_size]
        
        return self.state_predictor(state_vector).reshape(1, -1)

    def forward(
        self,
        task_name: str,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pooled_output: Optional[torch.Tensor] = None,
        discriminator: bool = False,
        output_hidden_states: bool = True,
        # Legacy compatibility
        src: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.
        
        Args:
            task_name: Name of task (lowercase)
            input_ids: Input IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length] (optional, for BERT)
            labels: Labels for training (optional)
            pooled_output: Pre-computed pooled output (for discriminator mode)
            discriminator: If True, use pooled_output directly
            output_hidden_states: If True, return all hidden states
            src: Legacy name for input_ids
            mask: Legacy name for attention_mask
            
        Returns:
            Tuple of (logits, features, model_state, pooled_output):
            - logits: Task predictions [batch_size, num_labels] or [batch_size, seq_len, vocab_size]
            - features: Hidden states from all layers (if output_hidden_states=True)
            - model_state: Model state for action predictor
            - pooled_output: Pooled encoder output
        """
        # Handle legacy parameter names
        if src is not None:
            input_ids = src
        if mask is not None:
            attention_mask = mask
        
        if discriminator:
            # Use pre-computed pooled output
            if self.task_type == 'language_modeling':
                raise ValueError("Discriminator mode not supported for language modeling")
            out = self.output_heads[task_name.lower()](pooled_output)
            if task_name == 'sts-b':
                out = nn.ReLU()(out)
            model_state = self._get_model_state()
            return (out, pooled_output, model_state, pooled_output)
        
        # Forward pass through model
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': output_hidden_states,
        }
        
        # Add token_type_ids only for encoder models
        if token_type_ids is not None and self.model_type == 'encoder':
            model_inputs['token_type_ids'] = token_type_ids
        
        outputs = self.model(**model_inputs)
        
        # Get appropriate output based on task type
        if self.task_type == 'language_modeling':
            # For LM: return logits directly
            logits = outputs.logits
            pooled_out = None  # No pooling for LM
        else:
            # For classification: get pooled output and apply task head
            pooled_out = self._get_pooled_output(outputs, attention_mask)
            logits = self.output_heads[task_name.lower()](pooled_out)
            
            # Apply ReLU for regression tasks
            if task_name == 'sts-b':
                logits = nn.ReLU()(logits)
        
        # Get model state
        model_state = self._get_model_state()
        
        # Extract features from hidden states
        features = None
        if output_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Get all hidden states except first (embeddings) and last
            hidden_states = outputs.hidden_states[1:-1]
            if hidden_states:
                # Stack and reshape
                features = torch.stack(hidden_states, dim=0)
                # Get first token representation for each layer
                if features.shape[2] > 0:  # Check sequence length
                    features = features[:, :, 0, :]  # [num_layers, batch_size, hidden_size]
        
        return (logits, features, model_state, pooled_out)
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory.
        
        Args:
            save_directory: Path to save directory
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base model
        self.model.save_pretrained(save_directory)
        
        # Save full model state dict (includes custom heads)
        torch.save(
            self.state_dict(),
            os.path.join(save_directory, 'mpdistil_model.bin')
        )
        
        # Save config
        self.config.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(
        cls, 
        load_directory: str, 
        tasks: list, 
        label_nums: Dict[str, int],
        task_type: str = 'classification'
    ):
        """Load model from directory.
        
        Args:
            load_directory: Path to load directory
            tasks: List of task names
            label_nums: Dict mapping task names to label counts
            task_type: 'classification' or 'language_modeling'
            
        Returns:
            Loaded FineTunedModel instance
        """
        import os
        
        # Load config
        config = AutoConfig.from_pretrained(load_directory)
        
        # Create model (this will load the base model)
        model = cls(tasks, label_nums, config, pretrained_model_name=load_directory, task_type=task_type)
        
        # Load full state dict if exists
        state_dict_path = os.path.join(load_directory, 'mpdistil_model.bin')
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        
        return model
