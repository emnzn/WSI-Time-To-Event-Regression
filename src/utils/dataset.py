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
        self.labels = self.generate_labels(label_dir)
        self.attenion_mil = attenion_mil
        self.pad = pad
        self.event = event
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
        and the associated Meningioma grade as the values.
        """

        labels = pd.read_csv(label_dir)

        labels = labels.to_dict()

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
        filename = self.filenames[idx]
        patient_id = Path(filename).stem

        event = self.labels[self.event][idx]
        time_to_event_key = f"time-to-{self.event}"
        time = self.labels[time_to_event_key][idx]

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
