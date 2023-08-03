# Paper Information

- **Paper Title**: Improving the Accuracy-Robustness Trade-off of Classifiers via Adaptive Smoothing.
- **Paper URL**: https://arxiv.org/abs/2301.12554. Note that the following results were achieved after the preprint paper was uploaded to ArXiv, and are thus not presented in the linked paper yet.
- **Paper Authors**: Yatong Bai, Brendon G Anderson, Aerin Kim, Somayeh Sojoudi.

# Leaderboard Claim(s)

## Model 1

- **Architecture**: ResNet-152 + WideResNet-70-16 + mixing network
- **Dataset**: CIFAR-100
- **Threat Model**: Linf
- **eps**: 8 / 255
- **Clean accuracy**: 85.21 %
- **Robust accuracy**: 38.72 %
- **Additional data**: True. The accurate base classifier was pre-trained on ImageNet-21k. Additionally, the robust base classifier used 50M synthetic images.
- **Evaluation method**: AutoAttack
- **Checkpoint and code**: The checkpoints and code are available [here](https://github.com/Bai-YT/AdaptiveSmoothing).

## Model 2

- **Architecture**: ResNet-152 + WideResNet-70-16 + mixing network
- **Dataset**: CIFAR-100
- **Threat Model**: Linf
- **eps**: 8 / 255
- **Clean accuracy**: 80.18 %
- **Robust accuracy**: 35.15 %
- **Additional data**: True. The accurate base classifier and the robust base classifier both used additional training data.
- **Evaluation method**: AutoAttack
- **Checkpoint and code**: The checkpoints and code are available [here](https://github.com/Bai-YT/AdaptiveSmoothing).

## Model 3

- **Architecture**: ResNet-152 + WideResNet-70-16 + mixing network
- **Dataset**: CIFAR-10
- **Threat Model**: Linf
- **eps**: 8 / 255
- **Clean accuracy**: 95.03 %
- **Robust accuracy**: 67.97 %
- **Additional data**: True. The accurate base classifier was pre-trained on ImageNet-21k. Additionally, the robust base classifier used 50M synthetic images.
- **Evaluation method**: AutoAttack
- **Checkpoint and code**: The checkpoints and code are available [here](https://github.com/Bai-YT/AdaptiveSmoothing).

# Model Zoo:

- [x] I want to add my models to the Model Zoo (check if true).
- [x] <del>I use an architecture that is included among those [here](https://github.com/RobustBench/robustbench/tree/master/robustbench/model_zoo/architectures) or in `timm`. If not,</del> I added the link to the architecture implementation so that it can be added.
- [x] I agree to release my model(s) under MIT license (check if true) **OR** under a custom license, located here: (put the custom license URL here if a custom license is needed. If no URL is specified, we assume that you are fine with MIT).
