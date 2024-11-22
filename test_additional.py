import pytest
import torch
from model import LightMNIST

def test_model_initialization():
    """Test 1: Verify model initialization"""
    model = LightMNIST()
    assert model is not None, "Model should be initialized"

def test_model_forward_pass():
    """Test 2: Verify forward pass"""
    model = LightMNIST()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"

def test_model_parameter_count():
    """Test 3: Verify parameter count is below 25,000"""
    model = LightMNIST()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"