"""
Кастомная loss функция "кнут и пряник" для адаптивного обучения
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class KnoutPryanikLoss(nn.Module):
    """
    Loss функция "кнут и пряник":
    - Пряник: снижение loss при правильном предсказании
    - Кнут: увеличение loss при ошибочном предсказании
    
    Особенно полезно для критичных классов (отсутствие деталей, серьезные вмятины)
    """
    
    def __init__(
        self,
        base_loss: nn.Module = None,
        reward_factor: float = 0.1,
        penalty_factor: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
        critical_classes: Optional[list] = None,
        temperature: float = 1.0
    ):
        """
        Args:
            base_loss: базовая loss функция (по умолчанию CrossEntropy)
            reward_factor: коэффициент снижения loss (пряник)
            penalty_factor: коэффициент увеличения loss (кнут)
            class_weights: веса классов для балансировки
            critical_classes: список критичных классов для усиленного кнута
            temperature: температурный параметр для softmax
        """
        super(KnoutPryanikLoss, self).__init__()
        
        self.base_loss = base_loss or nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.reward_factor = reward_factor
        self.penalty_factor = penalty_factor
        self.class_weights = class_weights
        self.critical_classes = critical_classes or [3]  # серьезные повреждения
        self.temperature = temperature
        
        # Статистика для адаптации
        self.correct_history = []
        self.class_error_counts = {}
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: логиты модели [batch_size, num_classes]
            targets: истинные метки [batch_size]
            
        Returns:
            dict с компонентами loss
        """
        batch_size = predictions.size(0)
        
        # Применяем температурное масштабирование
        scaled_predictions = predictions / self.temperature
        
        # Базовый loss
        base_loss_values = self.base_loss(scaled_predictions, targets)
        
        # Получаем предсказания
        pred_classes = torch.argmax(scaled_predictions, dim=1)
        
        # Создаем маски для правильных и неправильных предсказаний
        correct_mask = (pred_classes == targets).float()
        incorrect_mask = 1.0 - correct_mask
        
        # Вычисляем модификаторы loss
        reward_modifier = torch.ones_like(base_loss_values)
        penalty_modifier = torch.ones_like(base_loss_values)
        
        # Применяем пряник (снижение loss для правильных предсказаний)
        reward_modifier = reward_modifier - (correct_mask * self.reward_factor)
        
        # Применяем кнут (увеличение loss для неправильных предсказаний)
        penalty_modifier = penalty_modifier + (incorrect_mask * self.penalty_factor)
        
        # Усиленный кнут для критичных классов
        if self.critical_classes:
            critical_mask = torch.zeros_like(targets, dtype=torch.bool)
            for critical_class in self.critical_classes:
                critical_mask |= (targets == critical_class)
            
            critical_penalty = incorrect_mask * critical_mask.float() * self.penalty_factor
            penalty_modifier = penalty_modifier + critical_penalty
        
        # Комбинируем модификаторы
        final_modifier = reward_modifier * penalty_modifier
        
        # Применяем модификатор к loss
        modified_loss = base_loss_values * final_modifier
        
        # Статистика
        self._update_statistics(pred_classes, targets, correct_mask)
        
        return {
            'loss': modified_loss.mean(),
            'base_loss': base_loss_values.mean(),
            'reward_loss': (base_loss_values * reward_modifier).mean(),
            'penalty_loss': (base_loss_values * penalty_modifier).mean(),
            'accuracy': correct_mask.mean(),
            'critical_accuracy': self._calculate_critical_accuracy(pred_classes, targets)
        }
    
    def _update_statistics(self, predictions: torch.Tensor, targets: torch.Tensor, correct_mask: torch.Tensor):
        """Обновляет статистику для адаптивной настройки"""
        # История правильности
        self.correct_history.extend(correct_mask.cpu().numpy().tolist())
        if len(self.correct_history) > 1000:  # ограничиваем размер истории
            self.correct_history = self.correct_history[-1000:]
        
        # Подсчет ошибок по классам
        incorrect_predictions = predictions[correct_mask == 0]
        incorrect_targets = targets[correct_mask == 0]
        
        for target_class in incorrect_targets:
            class_id = target_class.item()
            if class_id not in self.class_error_counts:
                self.class_error_counts[class_id] = 0
            self.class_error_counts[class_id] += 1
    
    def _calculate_critical_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Вычисляет точность для критичных классов"""
        if not self.critical_classes:
            return torch.tensor(0.0, device=predictions.device)
        
        critical_mask = torch.zeros_like(targets, dtype=torch.bool)
        for critical_class in self.critical_classes:
            critical_mask |= (targets == critical_class)
        
        if critical_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        critical_predictions = predictions[critical_mask]
        critical_targets = targets[critical_mask]
        critical_accuracy = (critical_predictions == critical_targets).float().mean()
        
        return critical_accuracy
    
    def get_adaptive_factors(self) -> tuple:
        """Возвращает адаптивные коэффициенты на основе статистики"""
        if len(self.correct_history) < 100:
            return self.reward_factor, self.penalty_factor
        
        # Текущая точность
        recent_accuracy = np.mean(self.correct_history[-100:])
        
        # Адаптация коэффициентов
        if recent_accuracy > 0.9:  # модель слишком самоуверенная
            adaptive_penalty = self.penalty_factor * 1.2
            adaptive_reward = self.reward_factor * 0.8
        elif recent_accuracy < 0.6:  # модель плохо обучается
            adaptive_penalty = self.penalty_factor * 0.8
            adaptive_reward = self.reward_factor * 1.2
        else:
            adaptive_penalty = self.penalty_factor
            adaptive_reward = self.reward_factor
        
        return adaptive_reward, adaptive_penalty


class AdaptiveKnoutPryanikLoss(KnoutPryanikLoss):
    """
    Адаптивная версия KnoutPryanikLoss с автоматической настройкой коэффициентов
    """
    
    def __init__(self, *args, adaptation_frequency: int = 100, **kwargs):
        super(AdaptiveKnoutPryanikLoss, self).__init__(*args, **kwargs)
        self.adaptation_frequency = adaptation_frequency
        self.step_counter = 0
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass с адаптивной настройкой"""
        # Адаптируем коэффициенты
        if self.step_counter % self.adaptation_frequency == 0:
            self.reward_factor, self.penalty_factor = self.get_adaptive_factors()
        
        self.step_counter += 1
        
        return super().forward(predictions, targets)


class FocalKnoutPryanikLoss(nn.Module):
    """
    Комбинация Focal Loss и Knout Pryanik для работы с несбалансированными данными
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reward_factor: float = 0.1,
        penalty_factor: float = 0.3,
        class_weights: Optional[torch.Tensor] = None
    ):
        super(FocalKnoutPryanikLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reward_factor = reward_factor
        self.penalty_factor = penalty_factor
        self.class_weights = class_weights
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Комбинирует Focal Loss с механизмом кнута и пряника
        """
        # Вычисляем вероятности
        probs = F.softmax(predictions, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal Loss компонент
        focal_weight = (1 - target_probs) ** self.gamma
        focal_loss = -self.alpha * focal_weight * torch.log(target_probs + 1e-8)
        
        # Knout Pryanik компонент
        pred_classes = torch.argmax(predictions, dim=1)
        correct_mask = (pred_classes == targets).float()
        incorrect_mask = 1.0 - correct_mask
        
        # Модификаторы
        modifier = torch.ones_like(focal_loss)
        modifier = modifier - (correct_mask * self.reward_factor)  # пряник
        modifier = modifier + (incorrect_mask * self.penalty_factor)  # кнут
        
        # Финальный loss
        final_loss = focal_loss * modifier
        
        return {
            'loss': final_loss.mean(),
            'focal_loss': focal_loss.mean(),
            'accuracy': correct_mask.mean()
        }
