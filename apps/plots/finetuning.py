r"""
Functions to recover metrics related to finetuning runs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import logging
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from core.config import RESULT_DIR, SAVING_DIR
from core.utils import load_jsonl_to_numpy

logger = logging.getLogger("core")

# Paths
SAVE_DIR = SAVING_DIR / "runs"

# Trainable components in the ViT
VIT_COMPONENTS = ["LN1", "MHA", "LN2", "FC1", "FC2"]

# Trainable components in the ViT
VIT_COMPONENTS_MAP = {
    "all": "All",
    "attn_norm": "LN1",
    "mha": "MHA",
    "ffn_norm": "LN2",
    "ffn_fc1": "FC1",
    "ffn_fc2": "FC2",
}

# Learning rates
LR_VALUES = {
    "cifar10": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar100": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_contrast_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_gaussian_noise_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_motion_blur_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_snow_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "cifar10_c_speckle_noise_5": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "domainnet_clipart": ["3e-3", "1e-2", "3e-2", "6e-2"],
    "domainnet_sketch": ["3e-3", "1e-2", "3e-2", "6e-2"],
    "flowers102": ["1e-3", "3e-3", "1e-2", "3e-2"],
    "pet": ["1e-3", "3e-3", "1e-2", "3e-2"],
}

# Dataset names
DATASET_MAP = {
    "cifar10": "Cifar10",
    "cifar100": "Cifar100",
    "cifar10_c_contrast_5": "Contrast",
    "cifar10_c_gaussian_noise_5": "Gaussian Noise",
    "cifar10_c_motion_blur_5": "Motion Blur",
    "cifar10_c_snow_5": "Snow",
    "cifar10_c_speckle_noise_5": "Speckle Noise",
    "domainnet_clipart": "Clipart",
    "domainnet_sketch": "Sketch",
    "pet": "Pet",
    "flowers102": "Flowers102",
}


# ----------------------------------------------------------------------------
# Utils to aggregate results
# ----------------------------------------------------------------------------


def get_single_exp(dataset_name: str, seed: int, lr: str, comp: int, prefix: str = "vit") -> tuple:
    r"""Recover training and evaluation information for a given run."""

    # Get log_dir name
    log_dir = f"{prefix}_{dataset_name}_seed_{seed}_lr_{lr}_comp_{comp}"
    log_dir = SAVE_DIR / log_dir

    # Recover experiment configuration, model information and evaluation file
    with open(log_dir / "config.json") as f:
        exp_config = json.load(f)

    with open(log_dir / "metrics" / "info_model.jsonl") as f:
        info_model = json.load(f)

    with open(log_dir / "metrics" / "eval.jsonl") as f:
        eval_file = json.load(f)

    # Recover the training step of the checkpoint evaluated
    checkpoint_dir = Path(log_dir / "checkpoints")
    iterator = checkpoint_dir.iterdir()
    *_, last = iterator
    checkpoint_step = last.parts[-1]

    # Recover model information and evaluation results
    eval_data = {
        "dataset_name": dataset_name,
        "seed": int(seed),
        "max_n_steps": exp_config["n_steps"],
        "lr": float(lr),
        "trainable_components": "all",
        "model_size": info_model["model_params"],
        "n_step": checkpoint_step,
        "test_acc": eval_file["test_acc"],
    }

    # Recover runs
    data_keys = ["loss", "step", "grad_norm", "eval_loss", "eval_acc"]
    data = load_jsonl_to_numpy(log_dir / "metrics" / "raw_0.jsonl", keys=data_keys)

    # Index for training and evaluation
    not_training = np.isnan(data["loss"].astype(float))
    not_eval = np.isnan(data["eval_loss"].astype(float))

    # Recover training runs
    train_steps = data["step"][~not_training]
    train_loss = data["loss"][~not_training]
    grad_norms = data["grad_norm"][~not_training]
    training_runs = [train_steps, train_loss, grad_norms]

    # Recover validation runs
    val_steps = data["step"][~not_eval]
    val_loss = data["eval_loss"][~not_eval]
    val_acc = data["eval_acc"][~not_eval]
    validation_runs = [val_steps, val_loss, val_acc]

    return training_runs, validation_runs, eval_data


def get_evals_csv(dataset_name: str, seeds: list, lrs: list) -> None:
    r"""Recover and aggreate evaluation results for a given dataset."""
    all_results = []
    keys = [
        "dataset_name",
        "seed",
        "max_n_steps",
        "lr",
        "trainable_components",
        "model_size",
        "n_step",
        "test_acc",
    ]

    # Aggregate results for the finetuned models with no frozen component
    for seed in seeds:
        for lr in lrs:
            results = {}
            _, _, eval_data = get_single_exp(dataset_name=dataset_name, seed=seed, lr=lr, comp=0)
            for key in keys:
                results[key] = eval_data[key]
            all_results.append(results)

    # Save results
    df = pd.DataFrame(all_results)
    results_path = RESULT_DIR / "finetuning"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / f"{dataset_name}.csv"
    df.to_csv(path)


def get_runs(dataset_name: str, seeds: list, lrs: list) -> dict:
    r"""Recover the training and validation runs for a given dataset."""
    all_runs = {}

    # Aggregate results
    for lr in lrs:
        all_runs[lr] = {}
        for seed in seeds:
            all_runs[lr][seed] = {
                "model_size": None,
                "train_steps": None,
                "train_loss": None,
                "grad_norm": None,
                "val_steps": None,
                "val_loss": None,
                "val_acc": None,
            }
            training_runs, validation_runs, eval_data = get_single_exp(
                dataset_name=dataset_name, seed=seed, lr=lr, comp=0
            )
            if all_runs[lr][seed]["model_size"] is None:
                all_runs[lr][seed]["model_size"] = eval_data["model_size"]
                all_runs[lr][seed]["trainable_components"] = eval_data["trainable_components"]
            train_steps, train_loss, grad_norms = training_runs
            val_steps, val_loss, val_acc = validation_runs
            all_runs[lr][seed]["train_steps"] = train_steps
            all_runs[lr][seed]["train_loss"] = train_loss
            all_runs[lr][seed]["grad_norm"] = grad_norms
            all_runs[lr][seed]["val_steps"] = val_steps
            all_runs[lr][seed]["val_loss"] = val_loss
            all_runs[lr][seed]["val_acc"] = val_acc
    return all_runs


def get_data(dataset_name: str, folder: str) -> pd.DataFrame:
    r"""Load data from csv file."""
    path = RESULT_DIR / folder / f"{dataset_name}.csv"
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------------
# Results functions
# ----------------------------------------------------------------------------


def get_csv_results() -> None:
    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    seeds = [0, 42, 3407]
    for dataset_name in dataset_names:
        lrs = LR_VALUES[dataset_name]
        get_evals_csv(dataset_name=dataset_name, seeds=seeds, lrs=lrs)


def table_results(dataset_names: list, seeds: list) -> None:
    r"""Recover finetuning performance."""
    acc_mean = {}
    acc_std = {}

    for dataset_name in dataset_names:
        data = get_data(dataset_name, folder="finetuning")
        best_acc = 0
        std = 0
        for lr in LR_VALUES[dataset_name]:
            values = []
            for seed in seeds:
                root_ind = (
                    (data["lr"] == float(lr)) & (data["seed"] == int(seed)) & (data["trainable_components"] == "all")
                )
                test_acc = np.asarray(data[root_ind]["test_acc"])
                values.append(test_acc)
            std_temp = np.asarray(values).std()
            values = np.asarray(values).mean()
            if values > best_acc:
                best_acc = values
                std = std_temp
        acc_mean[dataset_name] = best_acc
        acc_std[dataset_name] = std

    print("Finetuning")
    for dataset_name in dataset_names:
        print(dataset_name)
        print(
            f"{np.round(acc_mean[dataset_name] * 100, 2)}",
            f"{np.round(acc_std[dataset_name] * 100, 2)}",
        )
        print("\n")


def get_table_results() -> None:
    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    seeds = [0, 42, 3407]
    table_results(dataset_names=dataset_names, seeds=seeds)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"csv": get_csv_results, "table": get_table_results})


# %% CLI
if __name__ == "__main__":
    main()
# %%
