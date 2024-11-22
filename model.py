import torch
import torch.nn as nn
import torch.nn.functional as F

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Reduced the size of fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 24)  # Reduced from 32 to 24
        self.fc2 = nn.Linear(24, 10)  # Reduced input from 32 to 24

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        torch.manual_seed(42)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)