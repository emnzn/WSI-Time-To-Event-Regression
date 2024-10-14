import os
from typing import Tuple, Union

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from torch.utils.tensorboard import SummaryWriter
from sksurv.metrics import concordance_index_censored

from utils import (
    WSIDataset, get_args, save_args, get_save_dirs,
    get_model, set_seed, ResNet, SwinTransformer,
    AttentionBasedMIL, get_training_checkpoint,
    RankDeepSurvLoss
)

def train(
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL],
    attention_mil: bool,
    device: str,
    grad_accumulation: int
    ) -> Tuple[float, float]:

    """
    Trains the model for one epoch.
    """

    metrics = {
        "running_loss": 0,
        "times": [],
        "events": [],
        "predictions": []
    }

    model.train()
    for i, (wsi_embedding, time, event, _) in enumerate(tqdm(dataloader, desc="Training in progress")):
        wsi_embedding = wsi_embedding.to(device)
        time = time.float().to(device)
        event = event.float().to(device)

        if attention_mil:
            pred, _ = model(wsi_embedding)

        else:
            pred = model(wsi_embedding)

        loss = criterion(pred, time, event)
        loss.backward()

        if (i + 1) % grad_accumulation == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["times"].extend(time.cpu().numpy())
        metrics["events"].extend(event.cpu().numpy())
        metrics["predictions"].extend(pred.squeeze().detach().cpu().numpy())

    times = np.array(metrics["times"])
    events = np.array(metrics["events"]).astype(bool)
    estimated_risk = -np.array(metrics["predictions"])

    epoch_loss = metrics["running_loss"] / len(dataloader)
    c_index = concordance_index(times, estimated_risk, events)

    return epoch_loss, c_index


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL],
    attention_mil: bool,
    device: str,
    ) -> Tuple[float, float]:

    """
    Runs validation for a single epoch.
    """

    metrics = {
        "running_loss": 0,
        "times": [],
        "events": [],
        "predictions": []
    }

    model.eval()
    for wsi_embedding, time, event, _ in tqdm(dataloader, desc="Validation in progress"):
        wsi_embedding = wsi_embedding.to(device)
        time = time.float().to(device)
        event = event.float().to(device)

        if attention_mil:
            pred, _ = model(wsi_embedding)

        else:
            pred = model(wsi_embedding)

        loss = criterion(pred, time, event)

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["times"].extend(time.cpu().numpy())
        metrics["events"].extend(event.cpu().numpy())
        metrics["predictions"].extend(pred.squeeze().detach().cpu().numpy())

    times = np.array(metrics["times"])
    events = np.array(metrics["events"]).astype(bool)
    estimated_risk = -np.array(metrics["predictions"])

    epoch_loss = metrics["running_loss"] / len(dataloader)
    c_index = concordance_index(times, estimated_risk, events)

    return epoch_loss, c_index


def main():
    config_dir = os.path.join("configs", "train-config.json")
    args = get_args(config_dir)
    attention_mil = True if args["model"] == "attention-mil" else False
    
    root_data_dir = os.path.join("..", "data", args["feature_extractor"], args["embedding_type"], f"split-{args['split_num']}")
    train_dir = os.path.join(root_data_dir, "train")
    val_dir = os.path.join(root_data_dir, "val")

    label_dir = os.path.join("..", "data", "labels.csv")
    model_dir, log_dir = get_save_dirs(args, mode="train")
    
    writer = SummaryWriter(log_dir)
    save_args(log_dir, args)
    set_seed(args["seed"])
    
    os.makedirs(model_dir, exist_ok=True)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps"

    train_dataset = WSIDataset(
        train_dir, 
        label_dir, 
        attention_mil, 
        args["pad"], 
        args["event"],
        args["augment"], 
        args["embedding_type"], 
        args["target_shape"]
        )
    
    val_dataset = WSIDataset(
        val_dir, 
        label_dir, 
        attention_mil, 
        args["pad"], 
        args["event"],
        False, 
        args["embedding_type"], 
        args["target_shape"]
        )

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    model, save_base_name = get_model(args)
    model = model.to(device)
    
    criterion = RankDeepSurvLoss(alpha=args["alpha"], beta=args["beta"])
    optimizer = optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

    min_val_loss = np.inf
    max_val_c_index = -np.inf

    for epoch in range(1, args["epochs"] + 1):
        writer.add_scalar("Learning Rate", scheduler.optimizer.param_groups[0]["lr"], epoch)
        print(f"Epoch [{epoch}/{args['epochs']}]")

        train_loss, train_c_index = train(
            dataloader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            attention_mil=attention_mil, 
            model=model, device=device, 
            grad_accumulation=args["grad_accumulation"]
            )

        print("\nTrain Statistics:")
        print(f"Loss: {train_loss:.4f} | C-Index: {train_c_index:.4f}\n")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/C-Index", train_c_index, epoch)

        val_loss, val_c_index = validate(
            dataloader=val_loader, 
            criterion=criterion, 
            model=model, 
            attention_mil=attention_mil, 
            device=device
            )
        
        print("\nValidation Statistics:")
        print(f"Loss: {val_loss:.4f} | C-Index: {val_c_index:.4f}\n")

        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/C-Index", val_c_index, epoch)

        if val_loss < min_val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, f"{save_base_name}-lowest-loss.pth"))
            min_val_loss = val_loss
            print("New minimum loss — model saved.")

        if val_c_index > max_val_c_index:
            torch.save(model.state_dict(), os.path.join(model_dir, f"{save_base_name}-highest-C-Index.pth"))
            max_val_c_index = val_c_index
            print("New maximum C-Index — model saved.")

        if epoch % 5 == 0:
            checkpoint = get_training_checkpoint(epoch, model, optimizer, scheduler)
            torch.save(checkpoint, os.path.join(model_dir, f"{save_base_name}-latest-checkpoint.pth"))
            print("Checkpoint saved.")

        scheduler.step()
        print("-------------------------------------------------------------\n")

    torch.save(checkpoint, os.path.join(model_dir, f"{save_base_name}-latest-checkpoint.pth"))


if __name__ == "__main__":
    main()