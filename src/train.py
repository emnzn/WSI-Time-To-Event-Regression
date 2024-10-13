import os
from typing import Tuple, Union, Dict, Any

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, f1_score

from utils import (
    WSIDataset, get_args, save_args, get_save_dirs,
    get_model, set_seed, ResNet, SwinTransformer,
    AttentionBasedMIL, get_training_checkpoint,
    get_criterion
)

def train(
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL],
    attenion_mil: bool,
    device: str,
    grad_accumulation: int
    ) -> Tuple[float, float, float]:

    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader: DataLoader
        The data loader to iterate over.

    criterion: nn.Module
        The loss function.

    optimizer: optim.Optimizer
        The optimizer for parameter updates.

    model: Union[ResNet, SwinTransformer, AttentionBasedMIL]
        The model to be trained.

    attenion_mil: bool
        Whether training under an Attention-based Multiple-Instance Learning scheme.
        This is used because the Attenion-Based MIL model returns the 
        attention weights placed on each instance.

    device: str
        One of [cuda, cpu].

    grad_accumulation: int
        The number of gradient accumulation steps before performing gradient descent.

    Returns
    -------
    epoch_loss: float
        The average loss for the given epoch.

    epoch_f1: float
        The average weighted f1 for the given epoch.

    epoch_balanced_accuracy: float
        The average balanced accuracy for the given epoch.    
    """

    metrics = {
        "running_loss": 0,
        "predictions": [],
        "targets": []
    }

    model.train()
    for i, (wsi_embedding, target, _) in enumerate(tqdm(dataloader, desc="Training in progress")):
        wsi_embedding = wsi_embedding.to(device)
        target = target.to(device)

        if attenion_mil:
            logits, _ = model(wsi_embedding)

        else:
            logits = model(wsi_embedding)

        loss = criterion(logits, target) / grad_accumulation
        loss.backward()

        if (i + 1) % grad_accumulation == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["predictions"].extend(pred.cpu().numpy())
        metrics["targets"].extend(target.cpu().numpy())

    epoch_loss = metrics["running_loss"] / (len(dataloader) / grad_accumulation)
    epoch_f1 = f1_score(metrics["targets"], metrics["predictions"], average="weighted")
    epoch_balanced_accuracy = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

    return epoch_loss, epoch_f1, epoch_balanced_accuracy


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL],
    attenion_mil: bool,
    device: str
    ) -> Tuple[float, float, float]:

    """
    Runs validation for a single epoch.
    """
    
    metrics = {
        "running_loss": 0,
        "predictions": [],
        "targets": []
    }

    model.eval()
    for wsi_embedding, target, _ in tqdm(dataloader, desc="Validation in progess"):
        wsi_embedding = wsi_embedding.to(device)
        target = target.to(device)

        if attenion_mil: 
            logits, _ = model(wsi_embedding)

        else: 
            logits = model(wsi_embedding)

        loss = criterion(logits, target)
        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["predictions"].extend(pred.cpu().numpy())
        metrics["targets"].extend(target.cpu().numpy())

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_f1 = f1_score(metrics["targets"], metrics["predictions"], average="weighted")
    epoch_balanced_accuracy = balanced_accuracy_score(metrics["targets"], metrics["predictions"])
    
    return epoch_loss, epoch_f1, epoch_balanced_accuracy


def main():
    config_dir = os.path.join("configs", "train-config.json")
    args = get_args(config_dir)
    attenion_mil = True if args["model"] == "attention-mil" else False
    
    root_data_dir = os.path.join("..", "data", args["feature_extractor"], args["embedding_type"], f"split-{args['split_num']}")
    train_dir = os.path.join(root_data_dir, "train")
    val_dir = os.path.join(root_data_dir, "val")

    label_dir = os.path.join("..", "data", "labels.csv")
    model_dir, log_dir = get_save_dirs(args, mode="train")
    
    writer = SummaryWriter(log_dir)
    save_args(log_dir, args)
    set_seed(args["seed"])
    
    os.makedirs(model_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = WSIDataset(train_dir, label_dir, attenion_mil, args["pad"], args["augment"], args["embedding_type"], args["target_shape"])
    val_dataset = WSIDataset(val_dir, label_dir, attenion_mil, args["pad"], False, args["embedding_type"], args["target_shape"])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    model, save_base_name = get_model(args)
    model = model.to(device)
    
    train_criterion, val_criterion = get_criterion(args)
    optimizer = optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

    min_val_loss = np.inf
    max_val_balanced_accuracy = -np.inf

    for epoch in range(1, args["epochs"] + 1):
        writer.add_scalar("Learning Rate", scheduler.optimizer.param_groups[0]["lr"], epoch)
        print(f"Epoch [{epoch}/{args['epochs']}]")

        train_loss, train_f1, train_balanced_accuracy = train(
            dataloader=train_loader, criterion=train_criterion, optimizer=optimizer, 
            attenion_mil=attenion_mil, model=model, device=device, grad_accumulation=args["grad_accumulation"]
            )

        print("\nTrain Statistics:")
        print(f"Loss: {train_loss:.4f} | F1 score: {train_f1:.4f} | Balanced Accuracy: {train_balanced_accuracy:.4f}\n")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/F1", train_f1, epoch)
        writer.add_scalar("Train/Balanced-Accuracy", train_balanced_accuracy, epoch)

        val_loss, val_f1, val_balanced_accuracy = validate(
            dataloader=val_loader, criterion=val_criterion, model=model, attenion_mil=attenion_mil, device=device
            )
        
        print("\nValidation Statistics:")
        print(f"Loss: {val_loss:.4f} | F1 Score: {val_f1:.4f} | Balanced Accuracy: {val_balanced_accuracy:.4f}\n")

        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/F1", val_f1, epoch)
        writer.add_scalar("Validation/Balanced-Accuracy", val_balanced_accuracy, epoch)

        if val_loss < min_val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, f"{save_base_name}-lowest-loss.pth"))
            min_val_loss = val_loss
            print("New minimum loss — model saved.")

        if val_balanced_accuracy > max_val_balanced_accuracy:
            torch.save(model.state_dict(), os.path.join(model_dir, f"{save_base_name}-highest-balanced-accuracy.pth"))
            max_val_balanced_accuracy = val_balanced_accuracy
            print("New maximum balanced accuracy — model saved.")

        if epoch % 5 == 0:
            checkpoint = get_training_checkpoint(epoch, model, optimizer, scheduler)
            torch.save(checkpoint, os.path.join(model_dir, f"{save_base_name}-latest-checkpoint.pth"))
            print("Checkpoint saved.")

        scheduler.step()
        print("-------------------------------------------------------------\n")

    torch.save(checkpoint, os.path.join(model_dir, f"{save_base_name}-latest-checkpoint.pth"))


if __name__ == "__main__":
    main()