#  Copyright 2024 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
import numpy as np
from time import time
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCELoss import nnUNetTrainerCELoss, nnUNetTrainerTopKLoss
from batchgenerators.utilities.file_and_folder_operations import join


class nnUNetTrainer_Loss_CE_checkpoints(nnUNetTrainerCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
    
    # def on_epoch_end(self):
    #     self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
    #     self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
    #     self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
    #     self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
    #                                            self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
    #     self.print_to_log_file(
    #         f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

    #     # handling periodic checkpointing
    #     current_epoch = self.current_epoch
    #     if (current_epoch + 1) % 100 == 0 and current_epoch != (self.num_epochs - 1):
    #         self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch+1}.pth'))

    #     if self.local_rank == 0:
    #         self.logger.plot_progress_png(self.output_folder)

    #     self.current_epoch += 1


class nnUNetTrainer_Loss_TopK_checkpoints(nnUNetTrainerTopKLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
    
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % 100 == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch+1}.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1