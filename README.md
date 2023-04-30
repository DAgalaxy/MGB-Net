# MGB-Net

## Usage

### 1. Get Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/R50-ViT-B_16.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv R50-ViT-B_16.npz ../model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz
```

### 2. Prepare data

The datasets we used are provided by TransUnet's authors. Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it(following the TransUnet's License).

### 3. Environment

environment with python=3.7, and then run "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24, you can reduce it to match GPU memory (please also decrease the base_lr linearly).

```bash
python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [EPSANet](https://github.com/murufeng/EPSANet)

## Citations
