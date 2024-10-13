import os
from typing import Dict, List

import numpy as np
import pandas as pd

def save_results(
    metrics: Dict[str, List[np.ndarray]], 
    save_dir: str,
    save_filename: str
    ):

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(save_dir, f"{save_filename}-results.csv"), index=False)