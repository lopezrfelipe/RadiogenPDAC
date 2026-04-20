from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
import numpy as np
import torch


class nnUNetTrainerCELoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = RobustCrossEntropyLoss(weight=None,
                                      ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerTopKLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = TopKLoss(weight=None, ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100, k=10)
        deep_supervision_scales = self._get_deep_supervision_scales()
        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerWCELoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        # higher weight assigned to class 1.
        loss = RobustCrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda(),
                                      ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    

class nnUNetTrainerCELoss2(nnUNetTrainer):
    # oversample positive cases (2x times)
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = RobustCrossEntropyLoss(weight=None,
                                      ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerCELossLesionSplit(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = RobustCrossEntropyLoss(weight=None,
                                      ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerCELossLesionSplit2(nnUNetTrainer):
    # oversample positive cases (2x times): new split
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = RobustCrossEntropyLoss(weight=None,
                                      ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
