"""Training logic for MPDistil.

This module implements the 4-phase training pipeline:
1. Teacher fine-tuning on main task
2. Student knowledge distillation with PKD
3. Meta-teacher learning (collaborative/competitive)
4. Curriculum learning with reinforcement learning
"""

from typing import Dict, Optional, Tuple
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy as cp
from tqdm import tqdm
import pandas as pd

from .metrics import compute_metrics, MetricCalculator
from .curriculum import RewardCalculator, CurriculumSampler, ActionSelector, discount_rewards
from .config import TrainingConfig, TaskConfig


class BasePhaseTrainer:
    """Base class for all training phases.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        device: Device (cuda/cpu)
        config: TrainingConfig
        task_config: TaskConfig
    """
    
    def __init__(
        self,
        model,
        optimizer,
        device: torch.device,
        config: TrainingConfig,
        task_config: TaskConfig
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.task_config = task_config
        
        # Loss function based on task type
        if task_config.is_regression:
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        task_name: str,
        split: str = 'eval'
    ) -> Tuple[list, Dict]:
        """Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader to evaluate on
            task_name: Name of task
            split: 'eval' or 'test'
            
        Returns:
            Tuple of (predictions, metrics_dict)
        """
        self.model.eval()
        
        all_labels = []
        all_preds = []
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if split == 'eval':
                    input_ids, attention_mask, token_type_ids, labels = batch
                    labels = labels.to(self.device)
                else:
                    input_ids, attention_mask, token_type_ids = batch
                
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                
                out = self.model(
                    src=input_ids,
                    task_name=task_name.lower(),
                    mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=False
                )
                
                student_out = out[0]
                
                # Get predictions
                if not self.task_config.is_regression:
                    student_out_ = student_out.argmax(-1)
                else:
                    student_out_ = student_out[:, 0].clip(0, 5)
                
                all_preds.extend(student_out_.cpu().numpy().tolist())
                
                if split == 'eval':
                    if not self.task_config.is_regression:
                        all_labels.extend(labels.cpu().numpy().astype(int).tolist())
                    else:
                        all_labels.extend(labels.cpu().numpy().tolist())
                    
                    # Compute loss
                    if self.task_config.is_regression:
                        loss = self.loss_fn(student_out.view(-1), labels.view(-1))
                    else:
                        loss = self.loss_fn(
                            student_out.view(-1, self.task_config.num_labels),
                            labels.view(-1)
                        )
                    
                    total_val_loss += loss.item() * labels.shape[0]
        
        # Compute metrics
        if split == 'eval':
            metrics = compute_metrics(
                task_name.lower(),
                all_preds,
                all_labels,
                self.task_config.metric
            )
            metrics['val_loss'] = total_val_loss / len(dataloader.dataset)
        else:
            metrics = {}
        
        metrics['task'] = task_name
        
        return all_preds, metrics
    
    def save_checkpoint(self, path: str, **extras):
        """Save checkpoint.
        
        Args:
            path: Path to save checkpoint
            **extras: Additional data to save
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **extras
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


class Phase1TeacherTrainer(BasePhaseTrainer):
    """Phase 1: Teacher fine-tuning on main task."""
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> Dict:
        """Train teacher model.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            
        Returns:
            Training history dict
        """
        best_score = 0
        history = {'train_loss': [], 'val_metrics': []}
        
        metric_calc = MetricCalculator(self.task_config.metric)
        
        for epoch in range(epochs):
            self.model.train()
            total_training_loss = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f'Phase 1 Epoch {epoch+1}/{epochs}', disable=not self.config.verbose)
            
            for batch in pbar:
                input_ids, attention_mask, token_type_ids, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                out = self.model.forward(
                    src=input_ids,
                    task_name=self.task_config.task_name.lower(),
                    mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                teacher_out = out[0]
                
                # Compute loss
                if self.task_config.is_regression:
                    loss = self.loss_fn(teacher_out.view(-1), labels.view(-1))
                else:
                    loss = self.loss_fn(
                        teacher_out.view(-1, self.task_config.num_labels),
                        labels.view(-1)
                    )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_training_loss += loss.item() * labels.shape[0]
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_training_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)
            
            # Evaluate
            preds, metrics = self.evaluate(val_loader, self.task_config.task_name, split='eval')
            history['val_metrics'].append(metrics)
            
            if self.config.verbose:
                print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Metrics={metrics}')
            
            # Save best model
            score = metric_calc.get_best_score(metrics)
            if score > best_score:
                best_score = score
                if self.config.save_checkpoints:
                    checkpoint_path = os.path.join(
                        self.config.output_dir,
                        'phase1',
                        f'teacher_{self.task_config.task_name}_{self.config.seed}.ckpt'
                    )
                    self.save_checkpoint(checkpoint_path, epoch=epoch, score=score)
        
        # Load best checkpoint
        if self.config.save_checkpoints:
            checkpoint_path = os.path.join(
                self.config.output_dir,
                'phase1',
                f'teacher_{self.task_config.task_name}_{self.config.seed}.ckpt'
            )
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
        
        return history


class Phase2PKDTrainer(BasePhaseTrainer):
    """Phase 2: Student knowledge distillation with Patient KD."""
    
    def __init__(
        self,
        model,
        optimizer,
        device,
        config,
        task_config,
        teacher_model
    ):
        super().__init__(model, optimizer, device, config, task_config)
        self.teacher_model = teacher_model
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> Dict:
        """Train student with knowledge distillation.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            
        Returns:
            Training history dict
        """
        best_score = 0
        history = {'train_loss': [], 'val_metrics': []}
        
        metric_calc = MetricCalculator(self.task_config.metric)
        
        for epoch in range(epochs):
            self.model.train()
            self.teacher_model.eval()
            
            total_training_loss = 0
            total_task_loss = 0
            
            pbar = tqdm(train_loader, desc=f'Phase 2 Epoch {epoch+1}/{epochs}', disable=not self.config.verbose)
            
            for batch in pbar:
                input_ids, attention_mask, token_type_ids, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Student forward
                out = self.model.forward(
                    src=input_ids,
                    task_name=self.task_config.task_name.lower(),
                    mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                student_out, _, _, student_pooler = out
                
                # Teacher forward (no grad)
                with torch.no_grad():
                    out = self.teacher_model.forward(
                        src=input_ids,
                        task_name=self.task_config.task_name.lower(),
                        mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    teacher_out = out[0]
                    teacher_pooler = out[1][:student_pooler.shape[0]] if out[1] is not None else out[3][:student_pooler.shape[0]]
                
                # Hard loss (task loss)
                if self.task_config.is_regression:
                    hard_loss = self.loss_fn(student_out.view(-1), labels.view(-1))
                else:
                    hard_loss = self.loss_fn(
                        student_out.view(-1, self.task_config.num_labels),
                        labels.view(-1)
                    )
                
                # Soft loss (distillation)
                if self.task_config.is_regression:
                    soft_loss = F.mse_loss(teacher_out, student_out)
                else:
                    T = self.config.temperature
                    soft_targets = F.softmax(teacher_out / T, dim=-1)
                    probs = F.softmax(student_out / T, dim=-1)
                    soft_loss = F.mse_loss(soft_targets, probs) * T * T
                
                # PKD loss (patient knowledge distillation)
                if self.config.beta == 0:
                    pkd_loss = torch.zeros_like(soft_loss)
                else:
                    t_features = teacher_pooler / teacher_pooler.norm(dim=-1).unsqueeze(-1)
                    s_features = student_pooler / student_pooler.norm(dim=-1).unsqueeze(-1)
                    pkd_loss = F.mse_loss(s_features, t_features, reduction="mean")
                
                # Total loss
                total_loss = (
                    self.config.alpha * soft_loss +
                    (1 - self.config.alpha) * hard_loss +
                    self.config.beta * pkd_loss
                )
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_training_loss += total_loss.item() * labels.shape[0]
                total_task_loss += hard_loss.item() * labels.shape[0]
                
                pbar.set_postfix({
                    'total_loss': total_loss.item(),
                    'task_loss': hard_loss.item()
                })
            
            avg_train_loss = total_training_loss / len(train_loader.dataset)
            avg_task_loss = total_task_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)
            
            # Evaluate
            preds, metrics = self.evaluate(val_loader, self.task_config.task_name, split='eval')
            history['val_metrics'].append(metrics)
            
            if self.config.verbose:
                print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Task Loss={avg_task_loss:.4f}, Val Metrics={metrics}')
            
            # Save best model
            score = metric_calc.get_best_score(metrics)
            if score > best_score:
                best_score = score
                if self.config.save_checkpoints:
                    checkpoint_path = os.path.join(
                        self.config.output_dir,
                        'phase2',
                        f'student_pkd_{self.task_config.task_name}_{self.config.seed}.ckpt'
                    )
                    self.save_checkpoint(checkpoint_path, epoch=epoch, score=score)
        
        # Load best checkpoint
        if self.config.save_checkpoints:
            checkpoint_path = os.path.join(
                self.config.output_dir,
                'phase2',
                f'student_pkd_{self.task_config.task_name}_{self.config.seed}.ckpt'
            )
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
        
        return history


class Phase3MetaTeacherTrainer(BasePhaseTrainer):
    """Phase 3: Meta-teacher learning with collaborative/competitive loss."""
    
    def __init__(
        self,
        model,
        optimizer,
        device,
        config,
        task_config,
        teacher_model,
        student_model
    ):
        super().__init__(model, optimizer, device, config, task_config)
        self.teacher_model = teacher_model
        self.student_model = student_model
    
    def train(
        self,
        held_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict:
        """Train meta-teacher.
        
        Args:
            held_loader: Held-out DataLoader
            val_loader: Validation DataLoader
            
        Returns:
            Training history dict
        """
        best_score = 0
        history = {'train_loss': []}
        
        metric_calc = MetricCalculator(self.task_config.metric)
        
        # Train for one epoch on held data
        self.model.train()
        self.teacher_model.eval()
        self.student_model.eval()
        
        total_meta_loss = 0
        
        pbar = tqdm(held_loader, desc='Phase 3: Meta-Teacher', disable=not self.config.verbose)
        
        for batch in pbar:
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Extract pooled outputs from frozen teacher and student
            with torch.no_grad():
                _, _, _, teacher_pooler_output = self.teacher_model(
                    src=input_ids,
                    task_name=self.task_config.task_name.lower(),
                    mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                _, _, _, student_pooler_output = self.student_model(
                    src=input_ids,
                    task_name=self.task_config.task_name.lower(),
                    mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            
            # Meta-teacher predictions on both
            teacher_out = self.model.forward(
                pooled_output=teacher_pooler_output,
                task_name=self.task_config.task_name.lower(),
                discriminator=True
            )[0]
            
            teacher_out2 = self.model.forward(
                pooled_output=student_pooler_output,
                task_name=self.task_config.task_name.lower(),
                discriminator=True
            )[0]
            
            # Compute loss
            if self.config.use_competitive_loss:
                # Competitive loss
                if self.task_config.is_regression:
                    teacher_loss2 = self.loss_fn(teacher_out.view(-1), labels.view(-1))
                else:
                    teacher_loss2 = self.loss_fn(
                        teacher_out.view(-1, self.task_config.num_labels),
                        labels.view(-1)
                    )
                
                # Competitive component
                if not self.task_config.is_regression:
                    teacher_out_prob = nn.Softmax(-1)(teacher_out).gather(
                        dim=1, index=labels.long().view(-1, 1)
                    ).squeeze()
                    teacher_out2_prob = nn.Softmax(-1)(teacher_out2).gather(
                        dim=1, index=labels.long().view(-1, 1)
                    ).squeeze()
                else:
                    teacher_out_prob = -1 * torch.abs(labels - teacher_out)
                    teacher_out2_prob = -1 * torch.abs(labels - teacher_out2)
                
                teacher_loss = -torch.mean(teacher_out_prob) + torch.mean(teacher_out2_prob) + teacher_loss2
            else:
                # Collaborative loss (Equation 2)
                if self.task_config.is_regression:
                    teacher_loss = (
                        0.5 * self.loss_fn(teacher_out.view(-1), labels.view(-1)) +
                        0.5 * self.loss_fn(teacher_out2.view(-1), labels.view(-1))
                    )
                else:
                    teacher_loss = (
                        0.5 * self.loss_fn(teacher_out.view(-1, self.task_config.num_labels), labels.view(-1)) +
                        0.5 * self.loss_fn(teacher_out2.view(-1, self.task_config.num_labels), labels.view(-1))
                    )
            
            teacher_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_meta_loss += teacher_loss.item()
            pbar.set_postfix({'meta_loss': teacher_loss.item()})
        
        avg_meta_loss = total_meta_loss / len(held_loader)
        history['train_loss'].append(avg_meta_loss)
        
        # Evaluate
        preds, metrics = self.evaluate(val_loader, self.task_config.task_name, split='eval')
        
        if self.config.verbose:
            print(f'Phase 3: Meta Loss={avg_meta_loss:.4f}, Val Metrics={metrics}')
        
        # Save checkpoint
        score = metric_calc.get_best_score(metrics)
        if self.config.save_checkpoints:
            loss_type = 'comp' if self.config.use_competitive_loss else 'col'
            checkpoint_path = os.path.join(
                self.config.output_dir,
                'phase3',
                f'teacher_meta_{loss_type}_{self.task_config.task_name}_{self.config.seed}.ckpt'
            )
            self.save_checkpoint(checkpoint_path, score=score)
        
        return history


class Phase4CurriculumTrainer(BasePhaseTrainer):
    """Phase 4: Curriculum learning with reinforcement learning."""
    
    def __init__(
        self,
        model,
        optimizer,
        device,
        config,
        task_config,
        action_model,
        action_optimizer,
        teacher_meta,
        task_loaders,
        label_nums
    ):
        super().__init__(model, optimizer, device, config, task_config)
        self.action_model = action_model
        self.action_optimizer = action_optimizer
        self.teacher_meta = teacher_meta
        self.task_loaders = task_loaders
        self.label_nums = label_nums
        
        # Initialize curriculum components
        task_names = list(task_loaders.keys())
        self.sampler = CurriculumSampler(task_names, task_loaders)
        self.action_selector = ActionSelector(action_model, task_names)
        self.reward_calculator = RewardCalculator(config.reward_type)
    
    def train(
        self,
        val_loader: DataLoader,
        num_episodes: int
    ) -> Dict:
        """Train with curriculum learning.
        
        Args:
            val_loader: Validation DataLoader
            num_episodes: Number of episodes
            
        Returns:
            Training history dict
        """
        best_score = 0
        all_rewards = []
        all_trajectories = []
        history = {'rewards': [], 'val_metrics': []}
        
        metric_calc = MetricCalculator(self.task_config.metric)
        
        held_loader_main = self.task_loaders[self.task_config.task_name]['held']['loader']
        
        for episode in range(num_episodes):
            trajectory = []
            batch_rewards = []
            batch_states = []
            batch_actions = []
            
            # Zero gradients at start of episode
            self.optimizer.zero_grad()
            self.action_optimizer.zero_grad()
            
            for step, batch_main in enumerate(held_loader_main):
                # Select initial task randomly
                if step == 0:
                    action_task = random.choice(list(self.task_loaders.keys()))
                    trajectory.append(action_task)
                
                # Sample batch from selected task
                batch = self.sampler.sample_batch(action_task)
                input_ids, attention_mask, token_type_ids, labels = batch
                
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Forward student
                student_out, _, state_space, student_pooler = self.model.forward(
                    src=input_ids,
                    task_name=action_task.lower(),
                    mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                num_labels = self.label_nums[action_task.lower()]
                
                # Forward teacher (no grad)
                with torch.no_grad():
                    teacher_out2 = self.teacher_meta.forward(
                        pooled_output=student_pooler,
                        task_name=action_task.lower(),
                        discriminator=True
                    )[0]
                
                # Compute loss
                is_regression = action_task.lower() in ['sts-b', 'stsb']
                if is_regression:
                    loss = nn.MSELoss()(student_out.view(-1), labels.view(-1))
                else:
                    loss = nn.CrossEntropyLoss()(student_out.view(-1, num_labels), labels.view(-1))
                
                # Add regularization from teacher
                if not is_regression:
                    student_prob = nn.Softmax(-1)(student_out).gather(
                        dim=1, index=labels.long().view(-1, 1)
                    ).squeeze()
                    teacher_prob = nn.Softmax(-1)(teacher_out2).gather(
                        dim=1, index=labels.long().view(-1, 1)
                    ).squeeze()
                else:
                    student_prob = -1 * torch.abs(labels - student_out)
                    teacher_prob = -1 * torch.abs(labels - teacher_out2)
                
                loss = loss - torch.mean(student_prob) + torch.mean(teacher_prob)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Select next action
                action_task, action_idx, _ = self.action_selector.select_action(state_space)
                trajectory.append(action_task)
                
                # Evaluate on main task to compute reward
                input_ids_main, attention_mask_main, token_type_ids_main, labels_main = batch_main
                input_ids_main = input_ids_main.to(self.device)
                attention_mask_main = attention_mask_main.to(self.device)
                token_type_ids_main = token_type_ids_main.to(self.device)
                labels_main = labels_main.to(self.device)
                
                with torch.no_grad():
                    student_out3 = self.model.forward(
                        src=input_ids_main,
                        task_name=self.task_config.task_name.lower(),
                        mask=attention_mask_main,
                        token_type_ids=token_type_ids_main,
                        output_hidden_states=False
                    )[0]
                    
                    teacher_out = self.teacher_meta.forward(
                        src=input_ids_main,
                        task_name=self.task_config.task_name.lower(),
                        mask=attention_mask_main,
                        token_type_ids=token_type_ids_main,
                        output_hidden_states=False
                    )[0]
                
                # Compute reward
                reward = self.reward_calculator.compute_reward(
                    student_out3,
                    teacher_out,
                    labels_main,
                    self.task_config.task_name,
                    self.task_config.is_regression
                )
                
                batch_rewards.append(reward)
                batch_states.append(state_space.detach().clone())
                batch_actions.append(action_idx.unsqueeze(0))
            
            # Update student
            self.optimizer.step()
            
            # Compute policy gradient loss (with fresh gradients)
            batch_states = torch.cat(batch_states, 0)
            action_tensor = torch.cat(batch_actions, 0).unsqueeze(1)
            reward_tensor = torch.FloatTensor(
                discount_rewards([r.item() for r in batch_rewards], self.config.gamma)
            ).to(self.device)
            
            policy_loss = self.action_selector.compute_policy_loss(
                batch_states,
                action_tensor,
                reward_tensor
            )
            
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), self.config.max_grad_norm)
            self.action_optimizer.step()
            
            # Track rewards
            total_reward = sum([r.item() for r in batch_rewards]) / len(held_loader_main.dataset)
            all_rewards.append(total_reward)
            all_trajectories.append(trajectory)
            
            if (episode + 1) % 10 == 0 or episode == 0:
                # Evaluate periodically
                preds, metrics = self.evaluate(val_loader, self.task_config.task_name, split='eval')
                history['val_metrics'].append(metrics)
                
                score = metric_calc.get_best_score(metrics)
                
                if self.config.verbose:
                    print(f'Episode {episode+1}/{num_episodes}: Reward={total_reward:.4f}, Val Metrics={metrics}')
                
                # Save best
                if score > best_score:
                    best_score = score
                    if self.config.save_checkpoints:
                        checkpoint_path = os.path.join(
                            self.config.output_dir,
                            'phase4',
                            f'best_student_{self.task_config.task_name}.ckpt'
                        )
                        self.save_checkpoint(checkpoint_path, episode=episode, score=score)
        
        history['rewards'] = all_rewards
        history['trajectories'] = all_trajectories
        
        # Load best checkpoint
        if self.config.save_checkpoints:
            checkpoint_path = os.path.join(
                self.config.output_dir,
                'phase4',
                f'best_student_{self.task_config.task_name}.ckpt'
            )
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
        
        return history


class MPDistilTrainer:
    """Orchestrates all 4 training phases.
    
    Args:
        teacher: Teacher model
        student: Student model
        action_model: Action predictor
        config: TrainingConfig
        task_config: TaskConfig
        device: Device
    """
    
    def __init__(
        self,
        teacher,
        student,
        action_model,
        config: TrainingConfig,
        task_config: TaskConfig,
        device: torch.device
    ):
        self.teacher = teacher
        self.student = student
        self.action_model = action_model
        self.config = config
        self.task_config = task_config
        self.device = device
    
    def run_all_phases(self, task_loaders: Dict, label_nums: Dict) -> Dict:
        """Execute all 4 training phases.
        
        Args:
            task_loaders: Task loaders dictionary
            label_nums: Label nums dictionary
            
        Returns:
            Complete training history
        """
        history = {}
        
        # Phase 1: Teacher Fine-tuning
        if self.config.teacher_epochs > 0 and not self.config.skip_teacher_training:
            print("\n=== Phase 1: Teacher Fine-tuning ===")
            t_optimizer = torch.optim.AdamW(
                self.teacher.parameters(),
                lr=self.config.teacher_lr,
                weight_decay=self.config.weight_decay
            )
            
            phase1_trainer = Phase1TeacherTrainer(
                self.teacher,
                t_optimizer,
                self.device,
                self.config,
                self.task_config
            )
            
            history['phase1'] = phase1_trainer.train(
                task_loaders[self.task_config.task_name]['train']['loader'],
                task_loaders[self.task_config.task_name]['eval']['loader'],
                self.config.teacher_epochs
            )
        
        # Phase 2: Student PKD
        print("\n=== Phase 2: Student Knowledge Distillation ===")
        s_optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.student_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Clone teacher for meta-learning
        teacher_model2 = cp(self.teacher)
        # Freeze the base model (encoder or decoder)
        for param in teacher_model2.model.parameters():
            param.requires_grad = False
        
        phase2_trainer = Phase2PKDTrainer(
            self.student,
            s_optimizer,
            self.device,
            self.config,
            self.task_config,
            teacher_model2
        )
        
        history['phase2'] = phase2_trainer.train(
            task_loaders[self.task_config.task_name]['train']['loader'],
            task_loaders[self.task_config.task_name]['eval']['loader'],
            self.config.student_epochs
        )
        
        # Phase 3: Meta-Teacher
        print("\n=== Phase 3: Meta-Teacher Learning ===")
        t_optimizer2 = torch.optim.AdamW(
            teacher_model2.parameters(),
            lr=self.config.meta_lr
        )
        
        phase3_trainer = Phase3MetaTeacherTrainer(
            teacher_model2,
            t_optimizer2,
            self.device,
            self.config,
            self.task_config,
            self.teacher,
            self.student
        )
        
        history['phase3'] = phase3_trainer.train(
            task_loaders[self.task_config.task_name]['held']['loader'],
            task_loaders[self.task_config.task_name]['eval']['loader']
        )
        
        # Phase 4: Curriculum Learning
        if self.config.num_episodes > 0 and len(task_loaders) > 1:
            print("\n=== Phase 4: Curriculum Learning ===")
            
            # Clone student for meta-learning
            student_model2 = cp(self.student)
            s_optimizer3 = torch.optim.AdamW(
                student_model2.parameters(),
                lr=self.config.student_lr
            )
            
            s_optimizer2 = torch.optim.AdamW(
                self.action_model.parameters(),
                lr=self.config.meta_lr
            )
            
            phase4_trainer = Phase4CurriculumTrainer(
                student_model2,
                s_optimizer3,
                self.device,
                self.config,
                self.task_config,
                self.action_model,
                s_optimizer2,
                teacher_model2,
                task_loaders,
                label_nums
            )
            
            history['phase4'] = phase4_trainer.train(
                task_loaders[self.task_config.task_name]['eval']['loader'],
                self.config.num_episodes
            )
            
            # Update student with best from phase 4
            self.student.load_state_dict(student_model2.state_dict())
        
        return history
