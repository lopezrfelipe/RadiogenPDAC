#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma (float): Focusing parameter. Default is 2.0.
            alpha (Tensor, optional): Class weights. Default is None.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]

        # Convert logits to probabilities
        log_probs = F.log_softmax(input, dim=1)
        probs = log_probs.exp()
        
        # Gather the probabilities of the true classes
        target = target.long()
        true_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        log_true_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

        # Compute the focal weight
        focal_weight = (1.0 - true_probs).pow(self.gamma)

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(input.device)
            focal_weight = focal_weight * alpha[target]

        # Compute the focal loss
        focal_loss = -focal_weight * log_true_probs

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class nnUNetTrainer_WFocalLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = FocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

