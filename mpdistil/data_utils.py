"""Data loading and preprocessing utilities for MPDistil.

This module provides functions to:
- Validate DataLoader batch formats
- Convert user DataLoaders to internal task_loaders structure
- Load SuperGLUE datasets from HuggingFace
- Create DataLoaders from custom data
"""

from typing import Dict, List, Tuple, Optional, Union
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
from transformers import AutoTokenizer


def validate_dataloader(
    loader: DataLoader,
    split: str = 'train',
    check_labels: bool = True
) -> bool:
    """Validate DataLoader batch format.
    
    Expected format:
    - train/val/held: (input_ids, attention_mask, token_type_ids, labels)
    - test: (input_ids, attention_mask, token_type_ids)
    
    Args:
        loader: DataLoader to validate
        split: Type of split ('train', 'val', 'test', 'held')
        check_labels: Whether to check for labels (False for test)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If batch format is incorrect
    """
    try:
        # Get first batch
        batch = next(iter(loader))
        
        if check_labels:
            if len(batch) != 4:
                raise ValueError(
                    f"Expected 4 elements in batch (input_ids, attention_mask, "
                    f"token_type_ids, labels), got {len(batch)}"
                )
            input_ids, attention_mask, token_type_ids, labels = batch
            
            # Check labels
            if not isinstance(labels, torch.Tensor):
                raise ValueError("Labels must be a torch.Tensor")
        else:
            if len(batch) != 3:
                raise ValueError(
                    f"Expected 3 elements in batch (input_ids, attention_mask, "
                    f"token_type_ids), got {len(batch)}"
                )
            input_ids, attention_mask, token_type_ids = batch
        
        # Check tensors
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("attention_mask must be a torch.Tensor")
        if not isinstance(token_type_ids, torch.Tensor):
            raise ValueError("token_type_ids must be a torch.Tensor")
        
        # Check shapes match
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have same shape")
        if input_ids.shape != token_type_ids.shape:
            raise ValueError("input_ids and token_type_ids must have same shape")
        
        return True
        
    except StopIteration:
        raise ValueError("DataLoader is empty")


def auto_detect_num_labels(dataloader: DataLoader) -> int:
    """Automatically detect number of labels from DataLoader.
    
    Args:
        dataloader: DataLoader with labels
        
    Returns:
        Number of unique labels
    """
    all_labels = []
    for batch in dataloader:
        labels = batch[-1]  # Last element is labels
        all_labels.extend(labels.cpu().numpy().tolist())
    
    return len(set(all_labels))


def split_dataloader(
    loader: DataLoader,
    split_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Split DataLoader into two parts.
    
    Args:
        loader: DataLoader to split
        split_ratio: Ratio for second split (default: 0.2 for held split)
        seed: Random seed
        
    Returns:
        Tuple of (main_loader, split_loader)
    """
    dataset = loader.dataset
    total_size = len(dataset)
    split_size = int(total_size * split_ratio)
    main_size = total_size - split_size
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create random indices
    indices = list(range(total_size))
    random.shuffle(indices)
    
    main_indices = indices[:main_size]
    split_indices = indices[main_size:]
    
    # Create subsets
    main_dataset = Subset(dataset, main_indices)
    split_dataset = Subset(dataset, split_indices)
    
    # Create new dataloaders
    main_loader = DataLoader(
        main_dataset,
        batch_size=loader.batch_size,
        sampler=RandomSampler(main_dataset),
        num_workers=0
    )
    
    split_loader = DataLoader(
        split_dataset,
        batch_size=loader.batch_size,
        sampler=RandomSampler(split_dataset),
        num_workers=0
    )
    
    return main_loader, split_loader


def prepare_task_loaders(
    main_task_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    meta_loaders: Optional[Dict[str, DataLoader]] = None,
    num_labels: Optional[int] = None,
    auto_split_held: bool = True,
    held_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """Convert user DataLoaders to internal task_loaders structure.
    
    Args:
        main_task_name: Name of main task
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Optional test DataLoader
        meta_loaders: Dict of auxiliary task DataLoaders {'TaskName': loader}
        num_labels: Number of labels (auto-detected if None)
        auto_split_held: Auto-split train data if meta_loaders is None
        held_ratio: Ratio for held split
        seed: Random seed
        
    Returns:
        Tuple of (task_loaders, label_nums)
        - task_loaders: Dict with structure required by trainer
        - label_nums: Dict mapping task names to label counts
    """
    # Validate main task loaders
    validate_dataloader(train_loader, split='train', check_labels=True)
    validate_dataloader(val_loader, split='val', check_labels=True)
    if test_loader is not None:
        validate_dataloader(test_loader, split='test', check_labels=False)
    
    # Auto-detect num_labels if not provided
    if num_labels is None:
        num_labels = auto_detect_num_labels(train_loader)
    
    # Create held loader
    if auto_split_held:
        train_loader, held_loader = split_dataloader(train_loader, held_ratio, seed)
    else:
        # Use validation as held
        held_loader = val_loader
    
    # Initialize task_loaders for main task
    task_loaders = {
        main_task_name: {
            'train': {
                'loader': train_loader,
                'dataset': train_loader.dataset
            },
            'held': {
                'loader': held_loader,
                'dataset': held_loader.dataset
            },
            'eval': {
                'loader': val_loader,
                'dataset': val_loader.dataset
            },
            'num_labels': num_labels
        }
    }
    
    # Add test loader if provided
    if test_loader is not None:
        task_loaders[main_task_name]['test'] = {
            'loader': test_loader,
            'dataset': test_loader.dataset
        }
    
    # Initialize label_nums
    label_nums = {main_task_name.lower(): num_labels}
    
    # Add meta loaders (auxiliary tasks)
    if meta_loaders is not None:
        for task_name, loader in meta_loaders.items():
            # Validate
            validate_dataloader(loader, split='held', check_labels=True)
            
            # Detect num_labels for this task
            task_num_labels = auto_detect_num_labels(loader)
            
            # Add to task_loaders (only held split for auxiliary tasks)
            task_loaders[task_name] = {
                'held': {
                    'loader': loader,
                    'dataset': loader.dataset
                },
                'num_labels': task_num_labels
            }
            
            label_nums[task_name.lower()] = task_num_labels
    
    return task_loaders, label_nums


class SimpleDataset(Dataset):
    """Simple dataset for tokenized data.
    
    Args:
        input_ids: List or tensor of input IDs
        attention_mask: List or tensor of attention masks
        token_type_ids: List or tensor of token type IDs
        labels: List or tensor of labels
    """
    
    def __init__(
        self,
        input_ids: Union[List, torch.Tensor],
        attention_mask: Union[List, torch.Tensor],
        token_type_ids: Union[List, torch.Tensor],
        labels: Union[List, torch.Tensor]
    ):
        self.input_ids = torch.tensor(input_ids) if not isinstance(input_ids, torch.Tensor) else input_ids
        self.attention_mask = torch.tensor(attention_mask) if not isinstance(attention_mask, torch.Tensor) else attention_mask
        self.token_type_ids = torch.tensor(token_type_ids) if not isinstance(token_type_ids, torch.Tensor) else token_type_ids
        self.labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx]
        )


def create_simple_dataloader(
    texts: Union[List[str], List[Tuple[str, str]]],
    labels: List[int],
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 128,
    batch_size: int = 8,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader from raw texts.
    
    Args:
        texts: List of strings or tuples (text_a, text_b) for pairs
        labels: List of integer labels
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader with correct batch format
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize
    if isinstance(texts[0], tuple):
        # Sentence pairs
        encodings = tokenizer(
            [t[0] for t in texts],
            [t[1] for t in texts],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    else:
        # Single sentences
        encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    # Create dataset
    dataset = SimpleDataset(
        input_ids=encodings['input_ids'],
        attention_mask=encodings['attention_mask'],
        token_type_ids=encodings['token_type_ids'],
        labels=labels
    )
    
    # Create dataloader
    sampler = RandomSampler(dataset) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle)
    )
    
    return loader


def load_superglue_dataset(
    task_name: str,
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 128,
    batch_size: int = 8,
    cache_dir: Optional[str] = None
) -> Tuple[Dict[str, DataLoader], int]:
    """Load SuperGLUE dataset from HuggingFace.
    
    Args:
        task_name: SuperGLUE task name ('CB', 'RTE', 'BoolQ', 'COPA', 'WiC', 'WSC')
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        batch_size: Batch size
        cache_dir: Cache directory for datasets
        
    Returns:
        Tuple of (loaders_dict, num_labels)
        - loaders_dict: Dict with 'train', 'val', 'test' DataLoaders
        - num_labels: Number of labels for the task
    """
    from datasets import load_dataset
    
    # Load dataset
    dataset = load_dataset('super_glue', task_name.lower(), cache_dir=cache_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Task-specific preprocessing
    task_configs = {
        'cb': {
            'text_cols': ['premise', 'hypothesis'],
            'label_col': 'label',
            'num_labels': 3
        },
        'rte': {
            'text_cols': ['premise', 'hypothesis'],
            'label_col': 'label',
            'num_labels': 2
        },
        'boolq': {
            'text_cols': ['question', 'passage'],
            'label_col': 'label',
            'num_labels': 2
        },
        'copa': {
            'text_cols': ['premise', 'choice1', 'choice2'],
            'label_col': 'label',
            'num_labels': 2
        },
        'wic': {
            'text_cols': ['sentence1', 'sentence2'],
            'label_col': 'label',
            'num_labels': 2
        },
        'wsc': {
            'text_cols': ['text'],
            'label_col': 'label',
            'num_labels': 2
        }
    }
    
    config = task_configs.get(task_name.lower())
    if config is None:
        raise ValueError(f"Task {task_name} not supported. Choose from: {list(task_configs.keys())}")
    
    def tokenize_function(examples):
        """Tokenize examples based on task configuration."""
        text_cols = config['text_cols']
        
        if len(text_cols) == 2:
            # Sentence pairs
            return tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        elif len(text_cols) == 1:
            # Single sentence
            return tokenizer(
                examples[text_cols[0]],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        else:
            # Multiple choice (like COPA) - simplified to first two
            return tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
    
    # Tokenize datasets
    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_val = dataset['validation'].map(tokenize_function, batched=True)
    
    # Convert to torch datasets
    tokenized_train.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'token_type_ids', config['label_col']]
    )
    tokenized_val.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'token_type_ids', config['label_col']]
    )
    
    # Create custom dataset wrapper
    class HFDatasetWrapper(Dataset):
        def __init__(self, hf_dataset, label_col):
            self.dataset = hf_dataset
            self.label_col = label_col
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            return (
                item['input_ids'],
                item['attention_mask'],
                item['token_type_ids'],
                item[self.label_col]
            )
    
    train_dataset = HFDatasetWrapper(tokenized_train, config['label_col'])
    val_dataset = HFDatasetWrapper(tokenized_val, config['label_col'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(val_dataset)
    )
    
    # Handle test set (no labels)
    if 'test' in dataset:
        tokenized_test = dataset['test'].map(tokenize_function, batched=True)
        tokenized_test.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'token_type_ids']
        )
        
        class HFTestDatasetWrapper(Dataset):
            def __init__(self, hf_dataset):
                self.dataset = hf_dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                return (
                    item['input_ids'],
                    item['attention_mask'],
                    item['token_type_ids']
                )
        
        test_dataset = HFTestDatasetWrapper(tokenized_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        test_loader = None
    
    loaders_dict = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return loaders_dict, config['num_labels']


def load_alpaca_dataset(
    tokenizer,
    max_seq_length: int = 512,
    batch_size: int = 4,
    num_samples: Optional[int] = None,
    dataset_name: str = 'tatsu-lab/alpaca'
) -> Dict:
    """Load Alpaca-style instruction tuning dataset for language modeling.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length (default: 512)
        batch_size: Batch size for dataloaders (default: 4)
        num_samples: Limit number of samples (for quick testing)
        dataset_name: HuggingFace dataset name (default: 'tatsu-lab/alpaca')
        
    Returns:
        Dict with 'train' and 'val' DataLoaders
    """
    from datasets import load_dataset
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name)
        if 'train' not in dataset:
            # If no train split, use the first available split
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name].train_test_split(test_size=0.1, seed=42)
    except Exception as e:
        print(f"Warning: Could not load {dataset_name}, using sample data. Error: {e}")
        # Create a small sample dataset for testing
        dataset = {
            'train': [
                {'instruction': 'What is AI?', 'input': '', 'output': 'AI stands for Artificial Intelligence.'},
                {'instruction': 'Explain machine learning', 'input': '', 'output': 'Machine learning is a subset of AI.'},
            ] * 10,
            'test': [
                {'instruction': 'What is deep learning?', 'input': '', 'output': 'Deep learning uses neural networks.'},
            ] * 5
        }
        from datasets import Dataset as HFDataset
        dataset = {
            'train': HFDataset.from_list(dataset['train']),
            'test': HFDataset.from_list(dataset['test'])
        }
    
    # Limit samples if requested
    if num_samples:
        dataset['train'] = dataset['train'].select(range(min(num_samples, len(dataset['train']))))
        if 'test' in dataset:
            dataset['test'] = dataset['test'].select(range(min(num_samples // 5, len(dataset['test']))))
    
    def tokenize_function(examples):
        """Format and tokenize instruction-following examples."""
        prompts = []
        for instruction, input_text, output in zip(
            examples.get('instruction', [''] * len(examples.get('output', []))),
            examples.get('input', [''] * len(examples.get('output', []))),
            examples.get('output', examples.get('text', [''] * len(examples)))
        ):
            if input_text and input_text.strip():
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            prompts.append(prompt)
        
        # Tokenize
        tokenized = tokenizer(
            prompts,
            max_length=max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For causal LM, labels = input_ids (shifted internally by model)
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    # Tokenize datasets
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    if 'test' in dataset:
        tokenized_val = dataset['test'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['test'].column_names
        )
    else:
        # Split train if no test set
        split_dataset = tokenized_train.train_test_split(test_size=0.1, seed=42)
        tokenized_train = split_dataset['train']
        tokenized_val = split_dataset['test']
    
    # Set format
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create wrapper dataset
    class AlpacaDatasetWrapper(Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            return (
                item['input_ids'],
                item['attention_mask'],
                torch.zeros_like(item['input_ids']),  # Dummy token_type_ids for compatibility
                item['labels']
            )
    
    train_dataset = AlpacaDatasetWrapper(tokenized_train)
    val_dataset = AlpacaDatasetWrapper(tokenized_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }
