import os
import json
from typing import Dict, Union


def get_args(arg_dir: str) -> Dict[str, Union[float, str]]:
    """
    Gets relevant arguments from a JSON file.

    Parameters
    ----------
    arg_dir: str
        The path to the JSON file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """
    
    with open(arg_dir, "r") as f:
        args = json.load(f)

    return args


def save_args(log_dir: str, args: Dict[str, Union[float, str]]):
    """
    Saves arguments inside a log directory.

    Parameters
    ----------
    log_dir: str
        The destination directory to save the arguments to.

    args: Dict[str, Union[float, str]]
        The arguments to be saved. The resulting JSON file will have a filename `run_config.json`.
    """

    path = os.path.join(log_dir, "run_config.json")

    organized_args = {
        "dataset": {
            "pad": args["pad"],
            "augment": args["augment"],
            "split_num": args["split_num"],
            "target_shape": args["target_shape"],
            "embedding_type": args["embedding_type"]
        },
        "training": {
            "seed": args["seed"],
            "epochs": args["epochs"],
            "eta_min": args["eta_min"],
            "batch_size": args["batch_size"],
            "num_classes": args["num_classes"],
            "learning_rate": args["learning_rate"],
            "feature_extractor": args["feature_extractor"],
            "loss_accumulation": args["loss_accumulation"]
        },
        "regularization": {
            "weight_decay": args["weight_decay"],
            "swin_dropout_probability": args["swin_dropout_probability"]
        },
        "model": {
            "model": args["model"],
            "resnet_normalization_method": args["resnet_normalization_method"]
        }
    }

    with open(path, "w") as f:
        json.dump(organized_args, f, indent=4)


def get_save_dirs(
    args: Dict[str, Union[float, str]],
    mode: str
    ):
    
    if mode == "train":
        if args["embedding_type"] == "isolated":
            if args["loss_accumulation"] > 1:
                model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "loss-accumulation", f"split-{args['split_num']}")
                log_dir = os.path.join("runs", args["embedding_type"], args["loss"], "loss-accumulation", f"split-{args['split_num']}", args["model"])

            else:
                model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "no-loss-accumulation", f"split-{args['split_num']}")
                log_dir = os.path.join("runs", args["embedding_type"], args["embedding_type"], "no-loss-accumulation", f"split-{args['split_num']}", args["model"])


        elif args["embedding_type"] == "stitched":
            if args["augment"]:
                model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "aug", f"split-{args['split_num']}")
                log_dir = os.path.join("runs", args["embedding_type"], args["loss"], "aug", f"split-{args['split_num']}", args["model"])

            else:
                model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "no-aug", f"split-{args['split_num']}")
                log_dir = os.path.join("runs", args["embedding_type"], args["loss"], "no-aug", f"split-{args['split_num']}", args["model"])


        return model_dir, log_dir
    
    if mode == "inference":
        if args["embedding_type"] == "isolated":
            if args["loss_accumulated"]:
                base_model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "loss-accumulation")
                base_save_dir = os.path.join("..", "assets", "inference-results", args["embedding_type"], "loss-accumulation")

            else:
                base_model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "no-loss-accumulation")
                base_save_dir = os.path.join("..", "assets", "inference-results", args["embedding_type"], args["loss"], "no-loss-accumulation")

        elif args["embedding_type"] == "stitched":
            if args["augmented"]:
                base_model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "aug")
                base_save_dir = os.path.join("..", "assets", "inference-results", args["embedding_type"], "aug")

            else:
                base_model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "no-aug")
                base_save_dir = os.path.join("..", "assets", "inference-results", args["embedding_type"], args["loss"], "no-aug")

        return base_model_dir, base_save_dir