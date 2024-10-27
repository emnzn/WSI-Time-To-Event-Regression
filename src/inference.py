import os
from typing import Tuple, Union

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

from utils import (
    WSIDataset, get_args, save_results, 
    get_save_dirs, get_model, ResNet, 
    SwinTransformer, BaseMIL, 
    AttentionBasedMIL, RankDeepSurvLoss,
    CoxPHLoss
)

@torch.no_grad()
def inference(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: Union[ResNet, SwinTransformer, BaseMIL, AttentionBasedMIL],
    attention_mil: bool,
    device: str,
    save_dir: str,
    save_filename: str
    ) -> Tuple[float, float]:

    """
    Runs inference.
    """

    accumulation_table = {
        "times": torch.tensor([], dtype=torch.float32).to(device),
        "events": torch.tensor([], dtype=torch.float32).to(device),
        "predictions": torch.tensor([], dtype=torch.float32).to(device)
    }

    model.eval()
    for wsi_embedding, time, event, _ in tqdm(dataloader, desc="Inference in progress"):
        wsi_embedding = wsi_embedding.to(device)
        time = time.float().to(device)
        event = event.float().to(device)

        if attention_mil:
            pred, _ = model(wsi_embedding)

        else:
            pred = model(wsi_embedding)

        accumulation_table["times"] = torch.cat([accumulation_table["times"], time])
        accumulation_table["events"] = torch.cat([accumulation_table["events"], event])
        accumulation_table["predictions"] = torch.cat([accumulation_table["predictions"], pred])

    time = accumulation_table["times"]
    event = accumulation_table["events"]
    pred = accumulation_table["predictions"]

    epoch_loss = criterion(pred, time, event)

    times = time.cpu().numpy()
    events = event.cpu().numpy().astype(bool)
    estimated_risk = pred.cpu().numpy()

    c_index = concordance_index(times, estimated_risk, events)

    metrics = {"times": times, "events": events, "predictions": "predictions"}
    save_results(metrics, save_dir, save_filename)

    return epoch_loss, c_index


def main():
    config_dir = os.path.join("configs", "inference-config.json")
    args = get_args(config_dir)
    attention_mil = True if args["model"] == "attention-mil" else False

    root_data_dir = os.path.join("..", "data", args["feature_extractor"], args["embedding_type"])
    base_model_dir, base_save_dir = get_save_dirs(args, mode="inference")
    num_splits = len(os.listdir(root_data_dir))

    losses = []
    c_indices = []

    for split_num in range(1, num_splits + 1):
        print(f"Trial [{split_num}/{num_splits}]")

        trial_dir = os.path.join(root_data_dir, f"split-{split_num}")
        inference_dir = os.path.join(trial_dir, "test")

        label_dir = os.path.join("..", "data", "labels.csv")
        model_dir = os.path.join(base_model_dir, f"split-{split_num}")
        save_dir = os.path.join(base_save_dir, f"split-{split_num}")
        os.makedirs(save_dir, exist_ok=True)
      
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inference_dataset = WSIDataset(
            inference_dir, 
            label_dir, 
            attention_mil, 
            args["pad"], 
            args["event"],
            args["augmented"], 
            args["embedding_type"], 
            args["target_shape"]
            )
        
        inference_loader = DataLoader(inference_dataset, batch_size=args["batch_size"], shuffle=False)

        model, save_base_name = get_model(args)
        model = model.to(device)

        weights_dir = os.path.join(model_dir, f"{save_base_name}-{args['weights']}")
        weights = torch.load(weights_dir, map_location=torch.device(device), weights_only=True)
        model.load_state_dict(weights)

        if args["loss"] == "rank-deep-surv":
            criterion = RankDeepSurvLoss(alpha=1.0, beta=1.0)
        

        elif args["loss"] == "cox-ph":
            criterion = CoxPHLoss()

        split_loss, split_c_index = inference(
            dataloader=inference_loader, 
            criterion=criterion, 
            attention_mil=attention_mil, 
            model=model, 
            device=device, 
            save_dir=save_dir,
            save_filename=save_base_name
            )

        losses.append(split_loss)
        c_indices.append(split_c_index)

        print("\nInference Statistics:")
        print(f"Loss: {split_loss:.4f} | C-Index: {split_c_index:.4f}\n")

        print("-------------------------------------------------------------\n")

    average_loss = sum(losses) / len(losses)
    average_c_index = sum(c_indices) / len(c_indices)
    c_index_std = np.std(c_indices)

    print("Summary:")
    print(f"Loss: {average_loss:.4f} | C-Index: {average_c_index:.4f} | C-Index STD: {c_index_std:.4f}\n")


if __name__ == "__main__":
    main()