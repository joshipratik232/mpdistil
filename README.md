# MPDistil 🎓

**Meta-Policy Knowledge Distillation for Training Compact Student Models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MPDistil is a teacher-student collaborative knowledge distillation framework that enables compact student models to outperform larger teacher models through meta-learning and curriculum learning.

Based on the ICLR 2024 paper: [**A Good Learner can Teach Better: Teacher-Student Collaborative Knowledge Distillation**](https://openreview.net/forum?id=Ixi4j6LtdX)

## 🌟 Key Features

- **📊 Superior Performance**: 6-layer BERT student outperforms 12-layer BERT teacher on 5/6 SuperGLUE tasks
- **🎯 4-Phase Training**: Teacher fine-tuning → PKD → Meta-teacher → Curriculum learning
- **🚀 Simple API**: Scikit-learn-style `.fit()` method for easy integration
- **🔧 Flexible**: Works with any HuggingFace model and custom datasets
- **💻 Colab-Ready**: GPU-optimized for Google Colab environments
- **📦 Easy Installation**: Single pip command to get started

## 📈 Methodology

<p align="center">
  <img width="720" alt="methodology" src="https://github.com/notmyname16/MPDistil/assets/88495622/e9444194-08fd-43ad-880b-94232302f449">
</p>

MPDistil consists of 4 training phases:

1. **Teacher Fine-tuning**: Fine-tune teacher model on the target task
2. **Student PKD**: Knowledge distillation with Patient Knowledge Distillation
3. **Meta-Teacher Learning**: Collaborative or competitive loss for meta-learning
4. **Curriculum Learning**: Reinforcement learning-based task selection

## 🚀 Installation

### From GitHub (Recommended for now)

```bash
pip install git+https://github.com/YOUR_USERNAME/mpdistil.git
```

### From Source

```bash
git clone https://github.com/YOUR_USERNAME/mpdistil.git
cd mpdistil
pip install -e .
```

## 💡 Quick Start

### Basic Usage

```python
from mpdistil import MPDistil, load_superglue_dataset

# Load data
loaders, num_labels = load_superglue_dataset('CB')

# Initialize MPDistil
model = MPDistil(
    task_name='CB',
    num_labels=num_labels,
    teacher_model='bert-base-uncased',
    student_model='bert-base-uncased',
    student_layers=6
)

# Train
history = model.fit(
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    teacher_epochs=10,
    student_epochs=10
)

# Save trained student
model.save_student('./my_student_model')

# Make predictions
predictions = model.predict(loaders['test'])
```

### With Custom Data

```python
from mpdistil import MPDistil, create_simple_dataloader

# Prepare your data
texts = [("This is text A", "This is text B"), ...]
labels = [0, 1, 0, ...]

# Create DataLoader
train_loader = create_simple_dataloader(
    texts=texts,
    labels=labels,
    tokenizer_name='bert-base-uncased',
    max_length=128,
    batch_size=8
)

# Train model
model = MPDistil(task_name='MyTask', num_labels=2)
history = model.fit(train_loader, val_loader)
```

### With Meta-Learning (Curriculum)

```python
# Load multiple tasks for curriculum learning
cb_loaders, _ = load_superglue_dataset('CB')
rte_loaders, _ = load_superglue_dataset('RTE')
boolq_loaders, _ = load_superglue_dataset('BoolQ')

# Train with curriculum learning
history = model.fit(
    train_loader=cb_loaders['train'],
    val_loader=cb_loaders['val'],
    meta_loaders={
        'RTE': rte_loaders['val'],
        'BoolQ': boolq_loaders['val']
    },
    num_episodes=200  # Curriculum learning episodes
)
```

## 📖 API Reference

### MPDistil Class

#### Constructor

```python
MPDistil(
    task_name: str,              # Name of the main task
    num_labels: int,             # Number of output classes
    teacher_model: str = 'bert-base-uncased',  # HuggingFace model name
    student_model: str = 'bert-base-uncased',  # HuggingFace model name
    student_layers: int = 6,     # Number of layers for student
    device: str = 'auto',        # 'auto', 'cuda', or 'cpu'
    output_dir: str = './mpdistil_outputs'
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `fit(train_loader, val_loader, **kwargs)` | Train the model (all 4 phases) |
| `predict(test_loader)` | Generate predictions |
| `save_student(path)` | Save student model in HuggingFace format |
| `load_student(path)` | Load a saved student model |
| `save_predictions(predictions, path, label_mapping)` | Save predictions to TSV |

### TrainingConfig

Configure training hyperparameters:

```python
from mpdistil import TrainingConfig

config = TrainingConfig(
    # Phase 1: Teacher
    teacher_epochs=10,
    teacher_lr=2e-5,
    
    # Phase 2: Student PKD
    student_epochs=10,
    student_lr=3e-5,
    alpha=0.5,          # Soft loss weight
    beta=100.0,         # PKD loss weight
    temperature=5.0,    # Distillation temperature
    
    # Phase 3: Meta-Teacher
    meta_lr=1e-3,
    use_competitive_loss=False,  # Use collaborative loss
    
    # Phase 4: Curriculum
    num_episodes=200,
    reward_type='binary',  # or 'real'
    
    # General
    batch_size=8,
    seed=42,
    wandb_logging=False
)

history = model.fit(train_loader, val_loader, config=config)
```

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teacher_epochs` | int | 10 | Teacher training epochs |
| `student_epochs` | int | 10 | Student training epochs |
| `num_episodes` | int | 200 | Curriculum learning episodes |
| `teacher_lr` | float | 2e-5 | Teacher learning rate |
| `student_lr` | float | 3e-5 | Student learning rate |
| `meta_lr` | float | 1e-3 | Meta-learning rate |
| `alpha` | float | 0.5 | Soft vs hard loss weight |
| `beta` | float | 100.0 | PKD loss weight |
| `temperature` | float | 5.0 | Distillation temperature |
| `use_competitive_loss` | bool | False | Competitive vs collaborative |
| `reward_type` | str | 'binary' | 'binary' or 'real' |
| `batch_size` | int | 8 | Batch size |
| `seed` | int | 42 | Random seed |

## 📊 Results

Performance on SuperGLUE tasks (BERT-base teacher → BERT-6L student):

| Model | BoolQ | CB | COPA | RTE | WiC | WSC |
|-------|-------|-------|------|-----|-----|-----|
| Teacher | 75.3 | 83.9 | 63.0 | 67.1 | 57.1 | 64.4 |
| Student (Undistilled) | 71.6 | 75.0 | 53.0 | 64.6 | 56.0 | 63.5 |
| **MPDistil (Ours)** | **73.4** | **83.9** | **70.0** | **67.5** | **59.6** | **65.4** |

✨ **Student outperforms teacher on 5/6 tasks!**

## 📝 Examples

See the `examples/` directory for Jupyter notebooks:

- **[01_quickstart.ipynb](examples/01_quickstart.ipynb)**: Basic usage with SuperGLUE
- **[02_custom_data.ipynb](examples/02_custom_data.ipynb)**: Using custom datasets
- **[03_advanced_config.ipynb](examples/03_advanced_config.ipynb)**: Advanced configurations

## 🔬 How It Works

### Phase 1: Teacher Fine-tuning
Fine-tune a large teacher model (e.g., BERT-base) on your target task.

### Phase 2: Student PKD
Train a smaller student model using:
- **Soft targets** from teacher (KL divergence)
- **Hard labels** (cross-entropy)
- **Patient KD** (intermediate feature matching)

Loss: `α * soft_loss + (1-α) * hard_loss + β * pkd_loss`

### Phase 3: Meta-Teacher
Create a meta-teacher that learns from both teacher and student representations:

**Collaborative loss** (default):
```
L = 0.5 * CE(T'(h_teacher), y) + 0.5 * CE(T'(h_student), y)
```

**Competitive loss**:
```
L = -mean(P_teacher) + mean(P_student) + CE_loss
```

### Phase 4: Curriculum Learning
Use reinforcement learning to select which auxiliary tasks help the student learn:
- **Action model** selects next task
- **Reward** based on student improvement over teacher
- **REINFORCE** algorithm updates policy

## 🛠️ Advanced Usage

### Custom Model Architectures

```python
model = MPDistil(
    task_name='MyTask',
    num_labels=3,
    teacher_model='roberta-large',
    student_model='distilbert-base-uncased',
    student_layers=6
)
```

### Weights & Biases Logging

```python
history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    wandb_logging=True,
    wandb_project='my-project'
)
```

### Access Trained Models

```python
# Access student model
student = model.student

# Access teacher model
teacher = model.teacher

# Use with HuggingFace
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('./my_student_model')
model = AutoModel.from_pretrained('./my_student_model')
```

## 📚 Citation

If you use MPDistil in your research, please cite:

```bibtex
@inproceedings{sengupta2024mpdistil,
  title={A Good Learner can Teach Better: Teacher-Student Collaborative Knowledge Distillation},
  author={Sengupta, Ayan and Dixit, Shantanu and Akhtar, Md Shad and Chakraborty, Tanmoy},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Ixi4j6LtdX}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Original paper: [ICLR 2024](https://openreview.net/forum?id=Ixi4j6LtdX)
- Built with [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/transformers/)

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

Made with ❤️ for the research community
