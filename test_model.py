import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import LightMNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    """Test 1: Verify model has less than 25000 parameters"""
    model = LightMNIST()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_accuracy():
    """Test 2: Verify model achieves >95% accuracy"""
    model = LightMNIST()
    model.load_state_dict(torch.load('mnist_model.pth'))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in train_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    
    accuracy = 100. * correct / total
    assert accuracy >= 95.0, f"Model accuracy is {accuracy}%, should be at least 95%"

def test_model_input_output():
    """Test 3: Verify model input/output shapes and values"""
    model = LightMNIST()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Model output shape incorrect"
    assert torch.isnan(output).sum() == 0, "Model produces NaN values"

def test_model_gradient_flow():
    """Test 4: Verify proper gradient flow during training"""
    model = LightMNIST()
    optimizer = optim.Adam(model.parameters())
    test_input = torch.randn(1, 1, 28, 28)
    test_target = torch.tensor([5])
    
    output = model(test_input)
    loss = F.nll_loss(output, test_target)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

def test_model_save_load():
    """Test 5: Verify model save and load functionality"""
    model = LightMNIST()
    test_input = torch.randn(1, 1, 28, 28)
    original_output = model(test_input)
    
    # Save and load
    torch.save(model.state_dict(), 'test_model.pth')
    loaded_model = LightMNIST()
    loaded_model.load_state_dict(torch.load('test_model.pth'))
    loaded_output = loaded_model(test_input)
    
    assert torch.allclose(original_output, loaded_output), "Model save/load failed"