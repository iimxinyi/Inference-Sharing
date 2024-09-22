# Inference-Sharing

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
```



