## Improving the Accuracy-Robustness Trade-off of Classifiers via Adaptive Smoothing

This repository is the code base for the preprint paper [Improving the Accuracy-Robustness Trade-off of Classifiers via Adaptive Smoothing](https://arxiv.org/abs/2301.12554).

We publically share two CIFAR-100 models that aim to defend the $\ell_\infty$ attack. Each of the proposed models rely on an accurate base classifier, a robust base classifier, and an optional "mixing network". The two proposed models share the same accurate base classifier but use two different robust base models and mixing networks.

### Running RobustBench to replicate the results

Running the [RobustBench](https://github.com/RobustBench/robustbench) benchmark should only require `pytorch`, `torchvision`, `numpy`, `click`, and `robustbench` packages.

Make a directory `<YOUR_MODEL_ROOT_DIR>` at a desired path to store the model checkpoints. Then, download the following models:
- Accurate base classifier: [Big Transfer (BiT)](https://github.com/google-research/big_transfer) ResNet-152 model finetuned on CIFAR-100 -- [download](https://drive.google.com/uc?export=download&id=1kdzhroeI9-pYuy0WQPF-DJH3-tDYJbvj)
- Robust base classifier 1: WideResNet-70-16 model from [this repo](https://github.com/wzekai99/DM-Improves-AT) -- [download](https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar100_linf_wrn70-16.pt) and rename as `cifar100_linf_edm_wrn70-16.pt`.
- Robust base classifier 2: WideResNet-70-16 model from [this repo](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness) -- [download](https://storage.googleapis.com/dm-adversarial-robustness/cifar100_linf_wrn70-16_with.pt) and rename as `cifar100_linf_trades_wrn70-16.pt`.
- Mixing network to be coupled with robust base classifier 1 -- [download](https://drive.google.com/uc?export=download&id=13busCr_xvU4i7jl8gc12VBfN6wRIjKM2)
- Mixing network to be coupled with robust base classifier 2 -- [download](https://drive.google.com/uc?export=download&id=1qP6v5XtbFoeaYp9BTzTmPVAjTKy6CYAf)

Now, organize `<YOUR_MODEL_ROOT_DIR>` following the structure below:
```
<YOUR_MODEL_ROOT_DIR>
│
└───Base
│   │   cifar100_linf_edm_wrn70-16.pt
│   │   cifar100_linf_trades_wrn70-16.pt
│   │   cifar100_bit_rn152.tar
│   
└───CompModel
    │   cifar-100_edm_best.pt
    │   cifar-100_trades_best.pt
```

To benchmark existing models with RobustBench, run the following:
```
python run_robustbench.py --root_dir <YOUR_MODEL_ROOT_DIR> --model_name {edm,trades}
```

Note that while the base classifiers may require additional (collected or synthesized) training data, the provided mixing networks were only trained on CIFAR-100 training data.

### Training a new model

To train a new model with the provided code, install the full environment. We require the following packages: `pytorch torchvision tensorboard pytorch_warmup numpy scipy matplotlib jupyter notebook ipykernel ipywidgets tqdm click PyYAML`.

To train, run the following:
```
python run.py --training --config configs/xxx.yaml
```

To evaluate, run the following:
```
python run.py --eval --config configs/xxx.yaml
```
