"""
distillation.py
Knowledge distillation loss: combines hard-label cross-entropy with a
soft-label KL-divergence term against a (larger, float32) teacher's output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        """
        alpha: weight on hard-label CE loss. (1 - alpha) weight on distillation term.
        temperature: softens both student and teacher logits before KL divergence.
                     Higher T -> softer distribution -> more "dark knowledge" transferred.
        """
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        hard_loss = self.ce(student_logits, targets)

        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        # KLDivLoss expects log-probabilities for input, probabilities for target.
        # Scale by T^2 as in the original Hinton et al. distillation paper, to
        # keep gradient magnitudes comparable across temperature choices.
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.T ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss