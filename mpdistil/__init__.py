"""MPDistil - Meta-Policy Knowledge Distillation

A teacher-student collaborative knowledge distillation framework
with curriculum learning for training compact student models.

Based on the ICLR 2024 paper:
"A Good Learner can Teach Better: Teacher-Student Collaborative Knowledge Distillation"

Example:
    >>> from mpdistil import MPDistil, load_superglue_dataset
    >>> 
    >>> # Load data
    >>> loaders, num_labels = load_superglue_dataset('CB')
    >>> 
    >>> # Initialize model
    >>> model = MPDistil(
    ...     task_name='CB',
    ...     num_labels=num_labels,
    ...     student_layers=6
    ... )
    >>> 
    >>> # Train
    >>> history = model.fit(
    ...     train_loader=loaders['train'],
    ...     val_loader=loaders['val'],
    ...     teacher_epochs=10,
    ...     student_epochs=10
    ... )
    >>> 
    >>> # Save
    >>> model.save_student('./my_student')
"""

from .core import MPDistil
from .config import TrainingConfig, TaskConfig
from .data_utils import (
    load_superglue_dataset,
    create_simple_dataloader,
    prepare_task_loaders
)
from .__version__ import __version__

__all__ = [
    'MPDistil',
    'TrainingConfig',
    'TaskConfig',
    'load_superglue_dataset',
    'create_simple_dataloader',
    'prepare_task_loaders',
    '__version__'
]
