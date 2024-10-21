import os
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset

class WSIDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        label_dir: str,
        attenion_mil: bool,
        pad: bool,
        event: str,
        augment: bool,
        embedding_type: str,
        target_shape: List[int]
        ):

        valid_events = ["recur", "death"]
        assert event in valid_events, f"event must be one of {valid_events}"

        self.data_dir = data_dir
        self.filenames = os.listdir(data_dir)
        self.event = event
        self.labels = self.generate_labels(label_dir)
        self.attenion_mil = attenion_mil
        self.pad = pad
        self.augment = augment
        self.embedding_type = embedding_type
        self.target_shape = target_shape

        self.transforms = A.Compose([
            A.Affine(
                translate_px=(-100, 100),
                rotate=0,
                shear=0,
                scale=1,
                cval=0
                )
            ])
        
        reference_ids = set([patient_id for _, patient_id in self.labels["id"].items()])
        assert all([Path(i).stem in reference_ids for i in self.filenames]), "All patient ids must have a label"

    def generate_labels(self, label_dir: str) -> Dict[str, str]:

        """
        Creates a dictionary containing the patient ids as keys
        and the associated time-to-event information as the values.
        """

        labels = pd.read_csv(label_dir)
        valid_ids = [Path(patient_id).stem for patient_id in self.filenames]
        valid_cols = ["id", self.event, f"time-to-{self.event}"]

        labels = labels[labels["id"].isin(valid_ids)][valid_cols]
        labels[f"time-to-{self.event}"] = labels[f"time-to-{self.event}"].map(lambda x: abs(x))
        data_dict = {col: [] for col in valid_cols}

        majority_class = labels[labels[self.event] == 0]
        minority_class = labels[labels[self.event] == 1]

        total_samples = len(labels)
        num_minority = len(minority_class)

        insertion_idx = total_samples // num_minority

        for i in range(total_samples):
            if i % insertion_idx == 0 and len(minority_class) > 0:
                sample = minority_class.iloc[-1]
                minority_class = minority_class[:-1]

                data_dict["id"].append(sample["id"])
                data_dict[self.event].append(sample[self.event])
                data_dict[f"time-to-{self.event}"].append(sample[f"time-to-{self.event}"])

            else:
                sample = majority_class.iloc[-1]
                majority_class = majority_class.iloc[:-1]

                data_dict["id"].append(sample["id"])
                data_dict[self.event].append(sample[self.event])
                data_dict[f"time-to-{self.event}"].append(sample[f"time-to-{self.event}"])

        labels = pd.DataFrame(data_dict)

        return labels
    
    def pad_embedding(
        self, 
        embedding: torch.Tensor, 
        target_shape: List[int]
        ) -> np.ndarray:

        """
        Pads the embedding to a target shape.
        The tensor must be of shape [C, H, W]
        """

        current_shape = embedding.shape[1:]

        delta_h = target_shape[0] - current_shape[0]
        delta_w = target_shape[1] - current_shape[1]

        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top
        
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left

        m = torch.nn.ZeroPad2d(padding=(pad_left, pad_right, pad_top, pad_bottom))

        padded_embedding = m(embedding)

        return padded_embedding
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        sample = self.labels.iloc[idx]
        patient_id = sample["id"]
        filename = f"{patient_id}.npy"

        event = sample[self.event]
        time_to_event_key = f"time-to-{self.event}"
        time = sample[time_to_event_key]

        embedding_path = os.path.join(self.data_dir, filename)

        if self.embedding_type == "isolated":
            embedding = torch.tensor(np.load(embedding_path))

        else:
            embedding = np.load(embedding_path)
            
            if self.augment:
                embedding = self.transforms(image=embedding)["image"]
            
            embedding = torch.tensor(embedding).permute(2, 0, 1) # [channels, height, width]

            if self.pad:
                embedding = self.pad_embedding(embedding, self.target_shape)

            if self.attenion_mil:
                channels, height, width = embedding.shape
                embedding = embedding.permute(1, 2, 0).reshape(height * width, channels)

        return embedding, time, event, patient_id
