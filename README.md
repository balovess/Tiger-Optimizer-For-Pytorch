[[中文](README_CN.md)|[English](README.md)]

# Tiger: A Budget-Conscious Neural Network Optimizer for PyTorch

Tiger is an optimizer designed for cost-conscious neural network training. It started as a TensorFlow project ([original repository](https://github.com/bojone/tiger/)), and this repository is a PyTorch adaptation of the original codebase.

## Features

- Achieves comparable performance to [AdamW](https://arxiv.org/abs/1711.05101) and [LAMB](https://arxiv.org/abs/1904.00962).
- Minimizes memory requirements when using gradient accumulation.
- Adaptive learning rates per parameter, similar to [LAMB](https://arxiv.org/abs/1904.00962).
- Simple strategy to prevent the model from collapsing to NaN.
- Can simulate any lr schedule with piecewise linear learning rates.

We would like to express our gratitude to the original TensorFlow project ([bojone/tiger](https://github.com/bojone/tiger/)) and its contributors for inspiring and providing the foundation for this PyTorch adaptation.
