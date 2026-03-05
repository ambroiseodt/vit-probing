r"""
Plotting functions related to linear probing runs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
import logging

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.config import FIGURE_DIR, RESULT_DIR, SAVING_DIR

logger = logging.getLogger("core")

# Paths
SAVE_DIR = SAVING_DIR / "probes"

# Trainable components in the ViT
VIT_COMPONENTS_MAP = {
    "attn_norm": "LN1",
    "attn": "MHA",
    "attn_res": "RC1",
    "ffn_norm": "LN2",
    "ffn_fc1": "FC1",
    "ffn_activation": "Act",
    "ffn_fc2": "FC2",
    "ffn_res": "RC2",
}

# Learning rates for the probing on finetuned models
LR_VALUES = {
    "cifar10": ["1e-2"],
    "cifar100": ["1e-2"],
    "cifar10_c_contrast_5": ["1e-2"],
    "cifar10_c_gaussian_noise_5": ["1e-2"],
    "cifar10_c_motion_blur_5": ["1e-2"],
    "cifar10_c_snow_5": ["1e-2"],
    "cifar10_c_speckle_noise_5": ["1e-2"],
    "domainnet_clipart": ["3e-2"],
    "domainnet_sketch": ["3e-2"],
    "flowers102": ["1e-2"],
    "pet": ["1e-2"],
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

# Figure golden ratio (from ICML style file)
WIDTH = 4
HEIGHT = WIDTH
FONTSIZE = 19
FONTSIZE_LEGEND = 17
LINEWIDTH = 5
ARROW_LINEWIDTH = 2.5
MARKER_SIZE = 500
ALPHA_GRID = 0.8
COLORS = {
    "LN1": "#daa4ac",
    "MHA": "#37abb5",
    "RC1": "#d0e2c2",
    "LN2": "#b153a1",
    "FC1": "#a291e1",
    "Act": "#96c0cf",
    "FC2": "#858ec2",
    "RC2": "#428379",
}

# Visual parameters
palette = sns.cubehelix_palette()
custom_params = {"axes.grid": False}
sns.set_theme(style="ticks", palette=palette, rc=custom_params)
sns.set_context("talk")
plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

# ----------------------------------------------------------------------------
# Utils to aggregate results
# ----------------------------------------------------------------------------


def get_single_exp(dataset_name: str, seed: int, lr: str, prefix: str = "vit", finetuned: bool = False) -> tuple:
    r"""Recover linear probing results for a given run."""

    corruption_dataset_map = {
        "cifar10_c_contrast_5": "cifar10_c-corruption-contrast-severity-5",
        "cifar10_c_gaussian_noise_5": "cifar10_c-corruption-gaussian_noise-severity-5",
        "cifar10_c_motion_blur_5": "cifar10_c-corruption-motion_blur-severity-5",
        "cifar10_c_snow_5": "cifar10_c-corruption-snow-severity-5",
        "cifar10_c_speckle_noise_5": "cifar10_c-corruption-speckle_noise-severity-5",
    }

    domainnet_dataset_map = {
        "domainnet_clipart": "domainnet-clipart",
        "domainnet_sketch": "domainnet-sketch",
    }

    # Get log_dir name
    if finetuned:
        log_dir = f"{prefix}_{dataset_name}_seed_{seed}_lr_{lr}_comp_0"
    else:
        if "cifar10_c" in dataset_name:
            dataset_name = corruption_dataset_map[dataset_name]
        elif "domainnet" in dataset_name:
            dataset_name = domainnet_dataset_map[dataset_name]
        log_dir = f"{prefix}_{dataset_name}_seed_0_pretrained"
    log_dir = SAVE_DIR / log_dir

    with open(log_dir / "linear_probing.json") as f:
        results_file = json.load(f)

    # Recover model information and evaluation results
    if finetuned:
        trainable_components = ["all"]
    else:
        trainable_components = ["none"]

    # Keep only the element
    trainable_components = trainable_components[0]
    meta_data = {
        "dataset_name": dataset_name,
        "trainable_components": trainable_components,
    }
    if finetuned:
        meta_data = meta_data | {"seed": int(seed), "lr": float(lr)}

    results = []
    for key in results_file.keys():
        block, comp = key.split("_", 1)
        block = block.split("block", 1)[-1]
        result = meta_data | {"block": block, "component": comp, "test_acc": results_file[key]}
        results.append(result)
    return results


def get_evals_csv(dataset_name: str, lrs: list) -> None:
    r"""Recover and aggreate linear probing results for a given dataset."""

    # Pretrained model
    all_results = get_single_exp(dataset_name=dataset_name, seed=None, lr=None, finetuned=False)
    df = pd.DataFrame(all_results)
    results_path = RESULT_DIR / "linear_probing" / "pretrained"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / f"{dataset_name}.csv"
    df.to_csv(path)

    # Finetuned model (full finetuning)
    all_results = []
    seed = 42
    for lr in lrs:
        all_results.extend(get_single_exp(dataset_name=dataset_name, seed=seed, lr=lr, finetuned=True))
    df = pd.DataFrame(all_results)
    results_path = RESULT_DIR / "linear_probing" / "finetuned"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / f"{dataset_name}.csv"
    df.to_csv(path)


def get_data(dataset_name: str, folder: str) -> pd.DataFrame:
    r"""Load data from csv file."""
    path = RESULT_DIR / folder / f"{dataset_name}.csv"
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------------


def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    """Save figure in pdf format."""
    figure_path = FIGURE_DIR
    if not figure_path.exists():
        figure_path.mkdir(parents=True, exist_ok=True)
    save_dir = figure_path / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def table_results(dataset_names: list) -> None:
    r"""Recover linear probing performance."""

    # Trainable components in the ViT
    components = ["attn_norm", "attn", "attn_res", "ffn_norm", "ffn_fc1", "ffn_activation", "ffn_fc2", "ffn_res"]

    # Results
    values = {}
    for dataset_name in dataset_names:
        latex_values = f"{DATASET_MAP[dataset_name]} & "
        for i, key in enumerate(components):
            # Recover linear probing performance over pretrained model
            linear_prob_pretrained = get_data(dataset_name, folder="linear_probing/pretrained")

            # Get results over the trainable component of the last layer
            root_ind = (linear_prob_pretrained["trainable_components"] == "none") & (
                linear_prob_pretrained["component"] == key
            )
            test_acc = linear_prob_pretrained[root_ind]["test_acc"].to_numpy() * 100
            best_values = test_acc.max()
            if i < 7:
                latex_values += f"{np.round(best_values, 2)} & "
            else:
                latex_values += f"{np.round(best_values, 2)}"
            if key in values:
                values[key].append(best_values)
            else:
                values[key] = [best_values]
        latex_values += r"\\"
        print(latex_values)
    latex_values = "Avg. &"
    for i, key in enumerate(values):
        if i < 7:
            latex_values += f"{np.round(np.mean(values[key]), 2)} &"
        else:
            latex_values += rf"{np.round(np.mean(values[key]), 2)} \\"
    print(latex_values)


def get_linear_probing_all(
    dataset_names: list,
    save: bool = False,
    ncol: int = 1,
) -> None:
    r"""Plot the linear probing performance over the depth."""
    nrows = 4
    ncols = 3
    width = 5
    height = 4
    figsize = (ncols * width, nrows * height)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 6)  # Use 6 units to allow half-column shifts

    # Validation loss for y-axis visualization
    acc_range = {
        "cifar10": [50, 75, 100],
        "cifar100": [25, 60, 95],
        "cifar10_c_contrast_5": [40, 70, 100],
        "cifar10_c_gaussian_noise_5": [30, 60, 90],
        "cifar10_c_motion_blur_5": [40, 68, 96],
        "cifar10_c_snow_5": [36, 66, 96],
        "cifar10_c_speckle_noise_5": [30, 60, 90],
        "domainnet_clipart": [4, 42, 80],
        "domainnet_sketch": [0, 35, 70],
        "flowers102": [14, 57, 100],
        "pet": [10, 53, 96],
    }

    n_layers = 12
    x_range = np.arange(n_layers) / (n_layers - 1) * 100
    colors = ["#c6a4e4", "#5b6da9"]

    # Results
    for i, dataset_name in enumerate(dataset_names):
        # Rows 1 & 1: Each plot takes 2 units (3 plots * 2 = 6 units)
        if i < 9:
            if i < 3:
                j = 0
            elif i >= 3 and i < 6:
                j = 1
                i -= 3
            elif i >= 6:
                j = 2
                i -= 6
            ax = fig.add_subplot(gs[j, i * 2 : (i + 1) * 2])
            if i + j == 0:
                ax_leg = ax

        # Row 3: Each plot takes 2 units, but we start at index 1 to center them
        # (1 + 2 + 2 + 1 = 6 units used, leaving 1 unit on each side)
        else:
            j = 3
            i -= 9
            ax = fig.add_subplot(gs[j, (i * 2) + 1 : (i * 2) + 3])

        # Recover linear probing performance over pretrained model
        linear_prob_pretrained = get_data(dataset_name, folder="linear_probing/pretrained")

        # Get results over the attention representation
        results_pretrained = []
        for k in range(n_layers):
            root_ind = (
                (linear_prob_pretrained["trainable_components"] == "none")
                & (linear_prob_pretrained["block"] == k)
                & (linear_prob_pretrained["component"] == "ffn_res")
            )
            test_acc_pretrained = linear_prob_pretrained[root_ind]["test_acc"].iloc[0]
            results_pretrained.append(test_acc_pretrained * 100)
        ax.plot(x_range, results_pretrained, label="pretrained", color=colors[0], lw=LINEWIDTH)

        # Recover linear probing performance over the fully finetuned model
        linear_prob_finetuned = get_data(dataset_name, folder="linear_probing/finetuned")

        # Get results over the attention representation
        results_finetuned = []
        for k in range(n_layers):
            root_ind = (
                (linear_prob_finetuned["trainable_components"] == "all")
                & (linear_prob_finetuned["block"] == k)
                & (linear_prob_finetuned["component"] == "ffn_res")
            )
            test_acc_finetuned = linear_prob_finetuned[root_ind]["test_acc"].iloc[0]
            results_finetuned.append(test_acc_finetuned * 100)
        ax.plot(
            x_range,
            results_finetuned,
            label="finetuned",
            color=colors[1],
            lw=LINEWIDTH,
            linestyle=(0, (5, 1)),
        )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=6)
        ax.set_xticks([0, 50, 100])

        # Fix y-axis ticks
        yticks = np.asarray(acc_range[dataset_name])
        ax.set_ylim(yticks.min(), yticks.max())
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.array(yticks, dtype=int))

        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        ax.set_ylabel("Accuracy (%)", fontsize=FONTSIZE)
        ax.set_title(f"{DATASET_MAP[dataset_name]} \n")
        sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    lines[0], lines[1] = lines[1], lines[0]
    labels[0], labels[1] = labels[1], labels[0]
    ax_leg.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.68, 0.44),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        handlelength=1.3,
        fontsize=FONTSIZE,
    )

    plt.tight_layout()
    if save:
        figname = "linear_probing_all"
        save_plot(figname=figname)
    plt.show()


def get_linear_probing(
    dataset_names: list,
    save: bool = False,
    ncol: int = 1,
) -> None:
    r"""Plot the linear probing performance over the depth."""
    ncols = len(dataset_names)
    width = 4
    height = width
    figsize = (ncols * width, height)
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Accuracy range for y-axis visualization
    acc_range = {
        "cifar10": [50, 75, 100],
        "cifar100": [25, 60, 95],
        "cifar10_c_contrast_5": [40, 70, 100],
        "cifar10_c_gaussian_noise_5": [30, 60, 90],
        "cifar10_c_motion_blur_5": [40, 68, 96],
        "cifar10_c_snow_5": [36, 66, 96],
        "cifar10_c_speckle_noise_5": [30, 60, 90],
        "domainnet_clipart": [4, 42, 80],
        "domainnet_sketch": [0, 35, 70],
        "flowers102": [14, 57, 100],
        "pet": [10, 53, 96],
    }

    n_layers = 12
    x_range = np.arange(n_layers) / (n_layers - 1) * 100
    colors = ["#c6a4e4", "#5b6da9"]

    # Results
    for i, dataset_name in enumerate(dataset_names):
        ax = axes[i]
        if i == 0:
            ax_leg = ax

        # Recover linear probing performance over pretrained model
        linear_prob_pretrained = get_data(dataset_name, folder="linear_probing/pretrained")

        # Get results over the transformer block output
        results_pretrained = []
        for k in range(n_layers):
            root_ind = (
                (linear_prob_pretrained["trainable_components"] == "none")
                & (linear_prob_pretrained["block"] == k)
                & (linear_prob_pretrained["component"] == "ffn_res")
            )
            test_acc_pretrained = linear_prob_pretrained[root_ind]["test_acc"].iloc[0]
            results_pretrained.append(test_acc_pretrained * 100)
        ax.plot(x_range, results_pretrained, label="pretrained", color=colors[0], lw=LINEWIDTH)

        # Recover linear probing performance over the fully finetuned model
        linear_prob_finetuned = get_data(dataset_name, folder="linear_probing/finetuned")

        # Get results over the transformer block output
        results_finetuned = []
        for k in range(n_layers):
            root_ind = (
                (linear_prob_finetuned["trainable_components"] == "all")
                & (linear_prob_finetuned["block"] == k)
                & (linear_prob_finetuned["component"] == "ffn_res")
            )
            test_acc_finetuned = linear_prob_finetuned[root_ind]["test_acc"].iloc[0]
            results_finetuned.append(test_acc_finetuned * 100)
        ax.plot(
            x_range,
            results_finetuned,
            label="finetuned",
            color=colors[1],
            lw=LINEWIDTH,
            linestyle=(0, (5, 1)),
        )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=6)
        ax.set_xticks([0, 50, 100])

        # Fix y-axis ticks
        yticks = np.asarray(acc_range[dataset_name])
        ax.set_ylim(yticks.min(), yticks.max())
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.array(yticks, dtype=int))

        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=FONTSIZE)
        ax.set_title(f"{DATASET_MAP[dataset_name]} \n")
        sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    lines[0], lines[1] = lines[1], lines[0]
    labels[0], labels[1] = labels[1], labels[0]
    ax_leg.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.68, 0.44),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        handlelength=1.3,
        fontsize=FONTSIZE,
    )

    plt.tight_layout()
    if save:
        figname = "linear_probing"
        save_plot(figname=figname)
    plt.show()


def get_linear_probing_components_all(
    dataset_names: list,
    save: bool = False,
    ncol: int = 1,
) -> None:
    r"""Plot the linear probing performance over the depth."""
    nrows = 4
    ncols = 3
    width = 5
    height = 4
    figsize = (ncols * width, nrows * height)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 6)  # Use 6 units to allow half-column shifts

    # Accuracy range for y-axis visualization
    acc_range = {
        "cifar10": [51, 73, 95],
        "cifar100": [25, 50, 75],
        "cifar10_c_contrast_5": [41, 62, 83],
        "cifar10_c_gaussian_noise_5": [32, 48, 64],
        "cifar10_c_motion_blur_5": [43, 58, 73],
        "cifar10_c_snow_5": [37, 54, 71],
        "cifar10_c_speckle_noise_5": [35, 50, 65],
        "domainnet_clipart": [6, 29, 52],
        "domainnet_sketch": [2, 20, 38],
        "flowers102": [14, 57, 100],
        "pet": [6, 51, 96],
    }

    n_layers = 12
    x_range = np.arange(n_layers) / (n_layers - 1) * 100

    # Trainable components in the ViT
    components = ["ffn_norm", "ffn_activation", "ffn_fc2", "ffn_res"]

    # Results
    for i, dataset_name in enumerate(dataset_names):
        # Rows 1 & 1: Each plot takes 2 units (3 plots * 2 = 6 units)
        if i < 9:
            if i < 3:
                j = 0
            elif i >= 3 and i < 6:
                j = 1
                i -= 3
            elif i >= 6:
                j = 2
                i -= 6
            ax = fig.add_subplot(gs[j, i * 2 : (i + 1) * 2])
            if i + j == 0:
                ax_leg = ax

        # Row 3: Each plot takes 2 units, but we start at index 1 to center them
        # (1 + 2 + 2 + 1 = 6 units used, leaving 1 unit on each side)
        else:
            j = 3
            i -= 9
            ax = fig.add_subplot(gs[j, (i * 2) + 1 : (i * 2) + 3])

        # Recover linear probing performance over pretrained model
        linear_prob_pretrained = get_data(dataset_name, folder="linear_probing/pretrained")

        # Get results over each hidden representation
        for key in components:
            trainable_component = VIT_COMPONENTS_MAP[key]
            linear_prob_pretrained = get_data(dataset_name, folder="linear_probing/pretrained")

            # Get results over the attention representation
            results_pretrained = []
            for k in range(n_layers):
                root_ind = (
                    (linear_prob_pretrained["trainable_components"] == "none")
                    & (linear_prob_pretrained["block"] == k)
                    & (linear_prob_pretrained["component"] == key)
                )
                test_acc_pretrained = linear_prob_pretrained[root_ind]["test_acc"].iloc[0]
                results_pretrained.append(test_acc_pretrained * 100)
            ax.plot(
                x_range,
                results_pretrained,
                color=COLORS[trainable_component],
                label=trainable_component,
                lw=LINEWIDTH,
            )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=6)
        ax.set_xticks([0, 50, 100])

        # Fix y-axis ticks
        yticks = np.asarray(acc_range[dataset_name])
        ax.set_ylim(yticks.min(), yticks.max())
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.array(yticks, dtype=int))

        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=FONTSIZE)
        ax.set_title(f"{DATASET_MAP[dataset_name]} \n")
        sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    lines, labels = lines[::-1], labels[::-1]
    ax_leg.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.78, 0.66),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        handlelength=1.3,
        fontsize=FONTSIZE,
    )

    plt.tight_layout()
    if save:
        figname = "linear_probing_components_all"
        save_plot(figname=figname)
    plt.show()


def get_linear_probing_components(
    dataset_names: list,
    save: bool = False,
    ncol: int = 1,
) -> None:
    r"""Plot the linear probing performance over the depth."""
    ncols = len(dataset_names)
    width = 4
    height = width
    figsize = (ncols * width, height)
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    # Accuracy range for y-axis visualization
    acc_range = {
        "cifar10": [51, 73, 95],
        "cifar100": [25, 50, 75],
        "cifar10_c_contrast_5": [41, 62, 83],
        "cifar10_c_gaussian_noise_5": [32, 48, 64],
        "cifar10_c_motion_blur_5": [43, 58, 73],
        "cifar10_c_snow_5": [37, 54, 71],
        "cifar10_c_speckle_noise_5": [35, 50, 65],
        "domainnet_clipart": [6, 29, 52],
        "domainnet_sketch": [2, 20, 38],
        "flowers102": [14, 57, 100],
        "pet": [6, 51, 96],
    }

    n_layers = 12
    x_range = np.arange(n_layers) / (n_layers - 1) * 100

    # Trainable components in the ViT
    components = ["ffn_norm", "ffn_activation", "ffn_fc2", "ffn_res"]

    # Results
    for i, dataset_name in enumerate(dataset_names):
        ax = axes[i]
        if i == 0:
            ax_leg = ax

        # Recover linear probing performance over pretrained model
        linear_prob_pretrained = get_data(dataset_name, folder="linear_probing/pretrained")

        # Get results over each hidden representation
        for key in components:
            trainable_component = VIT_COMPONENTS_MAP[key]
            linear_prob_pretrained = get_data(dataset_name, folder="linear_probing/pretrained")

            # Get results over the attention representation
            results_pretrained = []
            for k in range(n_layers):
                root_ind = (
                    (linear_prob_pretrained["trainable_components"] == "none")
                    & (linear_prob_pretrained["block"] == k)
                    & (linear_prob_pretrained["component"] == key)
                )
                test_acc_pretrained = linear_prob_pretrained[root_ind]["test_acc"].iloc[0]
                results_pretrained.append(test_acc_pretrained * 100)
            ax.plot(
                x_range,
                results_pretrained,
                color=COLORS[trainable_component],
                label=trainable_component,
                lw=LINEWIDTH,
            )

        # Visualization
        ax.grid(alpha=ALPHA_GRID, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", direction="out", length=6)
        ax.set_xticks([0, 50, 100])

        # Fix y-axis ticks
        yticks = np.asarray(acc_range[dataset_name])
        ax.set_ylim(yticks.min(), yticks.max())
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.array(yticks, dtype=int))

        ax.set_xlabel("Layer Depth (%)", fontsize=FONTSIZE)
        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=FONTSIZE)
        ax.set_title(f"{DATASET_MAP[dataset_name]} \n")
        sns.despine(fig, ax, trim=True, right=True, offset=10)

    # Common legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels, strict=False)]
    lines, labels = lines[::-1], labels[::-1]
    ax_leg.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.78, 0.66),
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        shadow=False,
        frameon=False,
        handlelength=1.3,
        fontsize=FONTSIZE,
    )

    plt.tight_layout()
    if save:
        figname = "linear_probing_components"
        save_plot(figname=figname)
    plt.show()


# ----------------------------------------------------------------------------
# Results functions
# ----------------------------------------------------------------------------


def get_csv_results() -> None:
    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_contrast_5",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    for dataset_name in dataset_names:
        lrs = LR_VALUES[dataset_name]
        get_evals_csv(dataset_name=dataset_name, lrs=lrs)


def get_table_results() -> None:
    dataset_names = [
        "cifar10",
        "cifar100",
        "cifar10_c_contrast_5",
        "cifar10_c_gaussian_noise_5",
        "cifar10_c_motion_blur_5",
        "cifar10_c_snow_5",
        "cifar10_c_speckle_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
        "flowers102",
        "pet",
    ]
    table_results(dataset_names=dataset_names)


def plot_figures() -> None:
    dataset_names = [
        "flowers102",
        "cifar10",
        "pet",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_motion_blur_5",
        "cifar100",
        "cifar10_c_speckle_noise_5",
        "cifar10_c_gaussian_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
    ]
    save = True
    get_linear_probing_all(dataset_names=dataset_names, save=save)

    dataset_names = [
        "flowers102",
        "cifar10",
        "cifar10_c_contrast_5",
        "cifar10_c_speckle_noise_5",
    ]
    save = True
    get_linear_probing(dataset_names=dataset_names, save=save)

    dataset_names = [
        "flowers102",
        "cifar10",
        "pet",
        "cifar10_c_contrast_5",
        "cifar10_c_snow_5",
        "cifar10_c_motion_blur_5",
        "cifar100",
        "cifar10_c_speckle_noise_5",
        "cifar10_c_gaussian_noise_5",
        "domainnet_clipart",
        "domainnet_sketch",
    ]
    save = True
    get_linear_probing_components_all(dataset_names=dataset_names, save=save)

    dataset_names = [
        "flowers102",
        "cifar10",
        "cifar10_c_contrast_5",
        "cifar10_c_speckle_noise_5",
    ]
    save = True
    get_linear_probing_components(dataset_names=dataset_names, save=save)


# %% Main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire({"csv": get_csv_results, "table": get_table_results, "plot": plot_figures})


# %% CLI
if __name__ == "__main__":
    main()
# %%
