import argparse
from typing import Callable, List
from tqdm import tqdm
import math
import os
from enum import Enum

import torch
from torch import nn

from mapillary_data_loader.load_mapillary import MapillaryDatasetPerceiver, MapillaryDatasetCNN
from perceiver import Perceiver, PerceiverClassifier
from metrics import compute_all_metrics

class ModelTrainer:
    """Class collecting all the functionality to train and evaluate a neural network model."""
    def __init__(self, loss_fn: Callable, optimizer: Callable, model: nn.Module, amp=True):
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._model = model.cuda()
        self._epoch_counter = 1

        # Mixed precision training
        self._scaler = torch.cuda.amp.GradScaler()
        self.train_step = self.train_step_amp if amp else self.train_step_fp
        if amp:
            def _foward_pass_amp(x_batch, y_batch):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.forward_pass(x_batch, y_batch)
            self._forward_pass = _foward_pass_amp
        else:
            self._forward_pass = forward_pass


    def forward_pass(self, x_batch, y_batch):
        """Forward pass and loss calculation."""
        # TODO Clean this up. Outputting a Dict in the dataloader would do.
        if len(x_batch) == 6:
            x, pe, orig_h, orig_w, scaled_h, scaled_w = x_batch
            x_batch = (x.to("cuda"), pe.to("cuda"), orig_h, orig_w, scaled_h, scaled_w)
        else:
            x, h, w = x_batch
            x_batch = (x.to("cuda"), h, w)
        y_batch = y_batch.to("cuda")

        pred = self._model(*x_batch)
        loss = self._loss_fn(pred, y_batch)
        return pred, loss

    def train_step_amp(self, x_batch, y_batch):
        """Apply a single training batch."""
        _, loss = self._forward_pass(x_batch, y_batch)

        # Backpropagation
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad(set_to_none=True)

        return loss

    def train_step_fp(self, x_batch, y_batch):
        loss = self._forward_pass(x_batch, y_batch)

        # Backpropagation
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss


    def train_epoch(self, dataloader):
        """A single training epoch."""
        total_train_loss = 0
        num_batches = 0
        self._model.train()
        for x_batch, y_batch in tqdm(dataloader):
            loss = self.train_step(x_batch, y_batch)

            # Call item to avoid having the pytorch loss with gradients on cpu.
            total_train_loss += loss.item()
            num_batches += 1
        return (total_train_loss/num_batches)

    @torch.no_grad()
    def eval_step(self, x_batch, y_batch):
        """Evaluation of a single batch."""
        return self._forward_pass(x_batch, y_batch)

    @torch.no_grad()
    def eval_epoch(self, dataloader):
        """A single evaluation epoch."""
        self._model.eval()

        # Note that the last batch might be smaller than the rest, so it is better to count the
        # number of individual samples.
        num_samples = 0
        total_eval_loss = 0
        predicted_classes = []
        true_classes = []
        heights = []
        widths = []
        with torch.no_grad():
            count = 0
            for x_batch, y_batch in tqdm(dataloader):
                pred, loss = self.eval_step(x_batch, y_batch)
                num_samples += len(y_batch)
                total_eval_loss += loss.item()*len(y_batch)

                # Aggregate predictions for metric calculation
                predicted_classes.append(torch.argmax(pred, dim=-1))
                true_classes.append(y_batch)
                
                # Extract heights and widths.
                height_index = 2 if len(x_batch) == 6 else 1
                width_index = 3 if len(x_batch) == 6 else 2
                heights.append(x_batch[height_index])
                widths.append(x_batch[width_index])

        avg_eval_loss = total_eval_loss/num_samples
        metric_dict = compute_all_metrics(predicted_classes, true_classes, heights, widths)    
        return avg_eval_loss, metric_dict

    def train_and_eval_epoch(self, train_loader, eval_loader):
        """Do a training and evaluation epoch and print information."""
        print("-"*100)
        print(f"Epoch {self._epoch_counter}")
        print("Train:")
        train_loss = self.train_epoch(train_loader)
        print(f"Train loss = {train_loss:.3f}")
        print("Eval:")
        eval_loss, eval_metrics = self.eval_epoch(eval_loader)
        print(f"Eval loss = {eval_loss:.3f}")
        print(f"Eval metrics = {eval_metrics}")
        self._epoch_counter += 1
        return train_loss, eval_loss, eval_metrics



class CallbackResult(Enum):
    """Results used in the convergence check."""
    NEW_BEST = 0
    WORSE = 1
    STOP = 2



class EarlyStopper:
    """Convergence criterion for model training.

    If the eval loss has not improved for a given number of training epochs, the training is
    considered to have congerged.
    """
    def __init__(self, tolerance: int):
        self._tolerance = tolerance
        self._fail_count = 0
        self._min_eval_loss = math.inf

    def __call__(self, new_eval_loss):
        if new_eval_loss < self._min_eval_loss:
            self._min_eval_loss = new_eval_loss
            self._fail_count = 0
            return CallbackResult.NEW_BEST
        if self._fail_count < self._tolerance:
            self._fail_count += 1
            return CallbackResult.WORSE
        return CallbackResult.STOP


if __name__ == '__main__':
    # Train TSPerceiver
    train_loader = MapillaryDatasetPerceiver(10000, True)
    eval_loader = MapillaryDatasetPerceiver(10000, False)
    perceiver = Perceiver(3, 100, 128, 1, 8, 6)
    classifier =  PerceiverClassifier(p, 10)
    out_dir = "trained_perceiver"
    train_model(train_loader, eval_loader, classifier, out_dir)
