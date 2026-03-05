# Vision Transformer Probing 
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](https://arxiv.org/abs/TBD)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/TBD)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-8C1515.svg)](https://openreview.net/forum?id=4lT3aScsRJ)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official implementation of  [Layer by layer, module by module: Choose both for optimal OOD probing of ViT](https://arxiv.org/pdf/TBD) (ICLR 2026 CAO Workshop). <br>
**Goal**: Analyze the linear probing of hidden representations across modules and layers in vision transformers. <br>
**Layer by layer**: In ID settings, final layers always yield better performance than intermediate layers. <br>
**Module by module**: In OOD settings, probing inputs and activations of intermediate feedforwards is better. <br>

## Abstract
> Recent studies have observed that intermediate layers of foundation models often yield more discriminative representations than the final layer. While initially attributed to autoregressive pretraining, this phenomenon has also been identified in models trained via supervised and discriminative self-supervised objectives. In this paper, we conduct a comprehensive study to analyze the behavior of intermediate layers in pretrained vision transformers. Through extensive linear probing experiments across a diverse set of image classification benchmarks, we find that distribution shift between pretraining and downstream data is the primary cause of performance degradation in deeper layers. Furthermore, we perform a fine-grained analysis at the level of modules. Our findings reveal that standard probing of transformer block outputs is suboptimal; instead, probing the activation within the feedforward network yields the best performance under significant distribution shift, whereas the normalized output of the multi-head self-attention module is optimal when the shift is weak.
<img width="1193" height="281" alt="ood_perf" src="https://github.com/user-attachments/assets/a0e5e3e6-c73b-4ed9-a771-041eeefce962" />
<img width="1193" height="455" alt="module" src="https://github.com/user-attachments/assets/6c0253c3-39bd-4013-88d7-9fd192f0262b" />

## Overview
Our codebase was tailored to study the probing of vision transformers under distribution shifts; we highly encourage you to use that as a template and modify it however you please to suit your experiments. We tried to make the code as easily modular as possible, so feel free to branch out or fork and play with it. Our codebase is structured as follows:

```
🛠️ vit-probing
┣ 📂apps 
┃ ┣ 📂vit # ViT probing
┃ ┃ ┣ 📂configs
┃ ┃ ┣ 📂scripts
┃ ┃ ┣ 📄eval.py
┃ ┃ ┣ 📄linear_probing.py
┃ ┃ ┣ 📄train.py
┃ ┃ ┗ 📄utils.py
┃ ┣ 📂plots # Figures
┗ 📂src 
  ┗ 📂core # Core library
    ┣ 📂data
    ┣ 📂model
    ┣ 📂monitor
    ┣ 📄__init__.py
    ┣ 📄config.py
    ┣ 📄distributed.py
    ┣ 📄optim.py
    ┗ 📄utils.py
```
The ```core``` folder contains essential and generic components related to vision transformers, which can be put together in the ```apps``` folder. In particular, ```apps/vit``` can be used to reproduce the experiments of our [paper](https://arxiv.org/pdf/TBD). 

## Getting started
The code runs Python 3.10+. Here are some installation instructions:
Install [miniforge](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Follow the instruction online, most likely you will execute the following commands:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash ~/Miniforge3-latest-Linux-x86_64.sh
source ~/.bashrc
```
Install Python in a new conda environment (be mindful to install a Python version compatible with Pytorch):
```bash
conda create -n myenv python==3.10
conda activate myenv
```
Install the repository (the vit dependencies are needed to use pretrained models):
```bash
git clone <repo url>
cd <repo path>
pip install -e ".[vit]"
```
To install the development and visualization dependencies, you can swap the previous command for the following one:
```bash
pip install -e ".[vit,dev,visu]"
```

#### Accelerate specific instructions
To load models from HuggingFace Transformers library, the accelerate package is needed. After installing it, it might be needed to configure it. Follow the instruction online [configure-accelerate](https://huggingface.co/docs/accelerate/en/basic_tutorials/install), most likely you will execute the following command and answer the questions prompted to you:
```bash
accelerate config
```

## Launching jobs
We provide below the commands useful to conduct experiments. They must be run from the root of the repository.  

### Configuration
Most experiments need a configuration file interfaced with the command line. Configuration objects are represented as dataclass objetc. 
For example, the file ```your_config.yaml``` looks like:
```yaml
log_dir: your_launch
model_name: base
patch_size: 16
dataset_name: cifar10
batch_size: 512
device: cuda:0
seed: 42
```
It can be used to initialize a dataclass that looks like
```python
@dataclass
class YourConfig:
  log_dir: str = "your_launch"
  model_name: str = "base"
  patch_size: int = 16
  dataset_name: str = "cifar10"
  batch_size: int = 512
  device: str = "cuda:0"
  seed: int = 42
```
In most scripts (```train.py```, ```eval.py```, ```linear_probing.py```), we use [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments). The behavior is as follows:
1. YourConfig is instantiated with its default values,
2. Those default values are overridden with the ones in your_config.yaml,
3. We override the result with the additional arguments provided through command line.
    

### Finetuning
To launch a finetuning job on Cifar10, run:
```bash
python -m apps.vit.train config=apps/vit/configs/cifar10.yaml
```
### Evaluation
To launch an evaluation job according to eval.yaml, run:
```bash
python -m apps.vit.eval config=apps/vit/configs/eval.yaml
```
### Linear probing
To launch a linear probing job according to linear_probing.yaml, run:
```bash
python -m apps.vit.linear_probing config=apps/vit/configs/linear_probing.yaml
```

## Reproducibility
The experiments of our [paper](https://arxiv.org/pdf/TBD) can be reproduced using the scripts in ```apps/vit/scripts```. Launching them will automatically create dedicated ```tmux``` sessions for each group of experiments. The finetuning experiments should be launched before the linear probing experiments since the latter depend on configuration files obtained after the finetuning runs such as the configuration files of finetuned models. After launching those scripts, the linear probing and finetuning performance can be recovered in a folder ```results/``` by running the following command from the root of the repository:
```bash
python -m apps.plots.finetuning csv
```
and
```bash
python -m apps.plots.linear_probing csv
```
The figures of our paper can then be reproduced using the files in ```apps/plots```.

## Acknowledgements
Our codebase is designed to study the hidden representation of transformers under distribition shifts. It draws inspiration from librairies like [vit-plasticity](https://github.com/ambroiseodt/vit-plasticity) and [lingua](https://github.com/facebookresearch/lingua).

## Contact
If you have any questions, feel free to reach out at [```ambroiseodonnattechnologie@gmail.com```](mailto:ambroiseodonnattechnologie@gmail.com).

## Citation
If you find our work useful, please consider giving a star ⭐, and citing us as:
```
@inproceedings{odonnat2026layer,
title={Layer by layer, module by module: Choose both for optimal {OOD} probing of ViT},
author={Ambroise Odonnat and Vasilii Feofanov and Laetitia Chapel and Romain Tavenard and Ievgen Redko},
booktitle={Catch, Adapt, and Operate: Monitoring ML Models Under Drift Workshop},
year={2026},
url={https://openreview.net/forum?id=4lT3aScsRJ}
}
```
