# MNIST Classification with PyTorch

[![Build Status](https://github.com/yourusername/your-repo-name/workflows/Test%20MNIST%20Model/badge.svg)](https://github.com/yourusername/your-repo-name/actions)

This project implements a lightweight MNIST classifier using PyTorch, achieving >95% accuracy in a single epoch with less than 25,000 parameters.

## Model Architecture
- 2 Convolutional layers
- 2 Max pooling layers
- 2 Fully connected layers
- Total parameters: ~24,000

## Features
- Efficient architecture with minimal parameters
- High accuracy in single epoch training
- Data augmentation with random rotation and translation
- Comprehensive test suite with GitHub Actions integration

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- pytest
- matplotlib
- numpy

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Train the model:
```bash
python train.py
```

2. Run tests:
```bash
pytest test_model.py -v
```

3. Generate augmented samples:
```bash
python augmentation.py
```

## Test Suite
The project includes automated tests for:
1. Parameter count verification (<25,000)
2. Model accuracy verification (>95%)
3. Input/output shape validation
4. Gradient flow verification
5. Model save/load functionality

## Data Augmentation
The training pipeline includes:
- Random rotation (±10 degrees)
- Random translation (±10%)

## License
MIT