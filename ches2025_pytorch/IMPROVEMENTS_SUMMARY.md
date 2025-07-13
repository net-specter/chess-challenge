# CNN-LSTM Model Improvements Summary

## Issues Fixed:

1. **Validation Accuracy Stagnation**: The validation accuracy was stuck at 0.2610 due to poor hyperparameters and lack of noise reduction.

## Algorithms and Techniques Added:

### 1. Noise Reduction Algorithms

- **Savitzky-Golay Filter**: Smooths the power traces while preserving important peaks and features

  - Window length: 11, Polynomial order: 3
  - Best for preserving signal characteristics while reducing noise

- **Gaussian Filter**: General noise reduction using Gaussian convolution

  - Kernel size: 5, Sigma: 1.0
  - Good for removing high-frequency noise

- **Median Filter**: Removes impulse noise and outliers

  - Kernel size: 5
  - Effective against salt-and-pepper noise

- **Adaptive Denoising**: Learnable noise reduction layer
  - Uses 1D convolution with trainable weights
  - Adapts to the specific noise characteristics of the dataset

### 2. Improved Activation Functions

- **GELU (Gaussian Error Linear Unit)**: Better gradient flow than ReLU
- **Swish**: Self-gated activation function (x \* sigmoid(x))
- These provide smoother gradients and better learning dynamics

### 3. Enhanced Hyperparameters

- **Batch sizes**: Changed to [64, 128, 256, 512] (better for deep learning)
- **Learning rates**: Increased range [1e-2, 5e-3, 1e-3, 5e-4, 1e-4] (faster convergence)
- **Optimizers**: Added AdamW with weight decay (better regularization)
- **CNN filters**: Increased to [32, 64, 128] (richer feature extraction)
- **CNN kernels**: Reduced to [8-16] (better local pattern detection)
- **LSTM hidden sizes**: Increased to [256, 512, 1024] (more capacity)
- **Dropout**: Optimized to [0.1, 0.2, 0.3] (better regularization)

### 4. Training Improvements

- **Learning Rate Scheduler**: ReduceLROnPlateau with patience=3
- **Weight Decay**: Added 1e-4 regularization to prevent overfitting
- **Batch Normalization**: Added to FC layers for training stability
- **Increased Epochs**: Changed from 5 to 20 for better convergence

### 5. Architecture Improvements

- **Noise Reduction First**: Applied before CNN feature extraction
- **Better Stride Handling**: Fixed stride calculations in CNN layers
- **Improved Pooling**: Better fallback values (2x2 instead of 1x1)
- **Enhanced FC Layers**: Added BatchNorm and reduced dropout

## Expected Results:

1. **Better Convergence**: Validation accuracy should now change and improve over epochs
2. **Noise Resilience**: Cleaner features extracted from noisy power traces
3. **Improved GE Performance**: Should achieve lower Guessing Entropy faster
4. **Stable Training**: Better gradient flow and training dynamics

## Key Algorithm References:

- **Savitzky-Golay Filter**: A. Savitzky and M. J. E. Golay (1964) - Smoothing and differentiation of data
- **GELU Activation**: Dan Hendrycks and Kevin Gimpel (2016) - Gaussian Error Linear Units
- **Swish Activation**: Prajit Ramachandran et al. (2017) - Searching for Activation Functions
- **AdamW Optimizer**: Ilya Loshchilov and Frank Hutter (2017) - Decoupled Weight Decay Regularization

The model should now show progressive improvement in validation accuracy instead of stagnation.
