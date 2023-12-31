import torch
import os
import json
from torch import nn
import multiprocessing as mp

from mapillary_data_loader.load_mapillary import get_perceiver_dataloader, get_cnn_dataloader
from mapillary_data_loader.make_class_list import mapillary_class_list
from perceiver import PerceiverClassifier, Perceiver
from training_loop import ModelTrainer, EarlyStopper, CallbackResult
from cnns import Resnet18, CNNClassifier


def train_model(train_loader, eval_loader, model, optimizer, output_path):
    """Train a neural network model until the convergence criterion is achieved."""
    # Make sure the output folder exists.
    os.makedirs(output_path, exist_ok=True)

    # Train the model until the eval loss stops improving for more than 7 epochs.
    trainer = ModelTrainer(nn.CrossEntropyLoss(), optimizer, model)
    callback = EarlyStopper(tolerance=7)
    metric_list = []
    while True:
        train_loss, eval_loss, eval_metrics = trainer.train_and_eval_epoch(train_loader, eval_loader)
        eval_metrics["train_loss"] = train_loss
        eval_metrics["eval_loss"] = eval_loss
        callback_result = callback(eval_loss)
        metric_list.append(eval_metrics)
        if callback_result == CallbackResult.NEW_BEST:

            # Save the model.
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, output_path + "model_checkpoint.pt")
            del checkpoint

            # Save the metrics
            with open(output_path + "metrics.json", "w") as f:
                json.dump(metric_list, f)

        elif callback_result == CallbackResult.STOP:
            break


def train_perceiver(train_loader, eval_loader):
    backbone = Perceiver(
        in_channels=3,
        n_latent=128, #512 in OG paper
        dim_latent=256, #1024 in OG paper
        n_heads_cross=1, #1 in OG paper
        n_heads_self=8, # 8 in OG paper
        n_self_per_cross=6, # 6 In OG paper
        n_blocks=6, # 6 in OG paper
        share_weights=True,
        fourier_pe=True,
        num_freq_bands=16, # 64 in OG paper
        max_freq=300,
        dropout_p=0.0
    )
    n_classes = len(mapillary_class_list())
    model = PerceiverClassifier(backbone, n_classes)
    #optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)

    return train_model(train_loader, eval_loader, model, optimizer, "perceiver_out/")


def train_resnet(train_loader, eval_loader):
    backbone = Resnet18(in_channels=3, channel_multiplier=1)
    n_classes = len(mapillary_class_list())
    model = CNNClassifier(backbone, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_model(train_loader, eval_loader, model, optimizer, "resnet_out/")


if __name__ == "__main__":
    # Avoid dataloading crash.
    torch.multiprocessing.set_sharing_strategy('file_system')

    #START_METHOD = "forkserver"
    #mp.set_start_method(START_METHOD)
    #mp.set_forkserver_preload(["torch", "torchvision"])

    # Empty GPU memory.
    torch.cuda.empty_cache()

    #perceiver_train_loader = get_perceiver_dataloader(batch_size=32, train=True, max_size=40000)
    #perceiver_eval_loader = get_perceiver_dataloader(batch_size=32, train=False, max_size=40000)
    #train_perceiver(perceiver_train_loader, perceiver_eval_loader)

    cnn_train_loader = get_cnn_dataloader(batch_size=32, train=True, im_size=(56, 56))
    cnn_eval_loader = get_cnn_dataloader(batch_size=32, train=False, im_size=(56, 56))
    train_resnet(cnn_train_loader, cnn_eval_loader)
