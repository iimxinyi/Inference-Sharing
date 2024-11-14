# Inference-Sharing

**Target:** The aim is to explore the misalignment between users' intentions and the generated contents, with a focus on guiding the design of an efficient hybrid inference scheme.

**Paper:** "QoS-Driven Hybrid Inference Scheme for Generative Diffusion Models in MEC-Enabled AI-Generated Content Networks."  --Submitted to ICC 2025

## Environment Setup

Create a new conda environment.

```shell
conda create --name LVM python==3.10
```

## Activate Environment

Activate the created environment.

```shell
conda activate LVM
```

## Install Required Packages

ubuntu==20.04, cuda==11.8
```shell
pip install sentence-transformers==3.1.1
pip install --upgrade diffusers transformers
pip install torch==2.4.1
pip install accelerate==0.34.2
pip install protobuf==5.25.2
pip install sentencepiece==0.2.0
pip install openai-clip==1.0.1
pip install torchvision==0.19.1
pip install openpyxl==3.1.5
```



