[[中文](README.md)|English]

# Tiger

A **Tig**ht-fisted Optimiz**er**, an optimizer that is extremely budget-conscious!

## Features

- Achieves comparable performance to [AdamW](https://arxiv.org/abs/1711.05101) and [LAMB](https://arxiv.org/abs/1904.00962).
- Minimizes memory requirements when using gradient accumulation.
- Adaptive learning rates per parameter, similar to [LAMB](https://arxiv.org/abs/1904.00962).
- Simple strategy to prevent the model from collapsing to NaN.
- Can simulate any lr schedule with piecewise linear learning rates.

