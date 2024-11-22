import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

def get_augmentation_pipeline():
    """Define the augmentation pipeline"""
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def show_augmented_samples(dataset, num_samples=5):
    """Display original and augmented versions of some samples"""
    # Set up the figure
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    # Basic transform for original images
    basic_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Augmentation transform
    augmentation = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])

    for i in range(num_samples):
        # Get a sample
        img, label = dataset[i]
        
        # Convert tensor to PIL Image for augmentation
        img_pil = transforms.ToPILImage()(img)
        
        # Apply augmentation
        aug_img = augmentation(img_pil)
        
        # Convert both to tensors for display
        img_display = torch.tensor(np.array(img_pil))
        aug_img_display = torch.tensor(np.array(aug_img))

        # Display original
        axes[0, i].imshow(img_display.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {label}')

        # Display augmented
        axes[1, i].imshow(aug_img_display.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmented')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = MNIST('./data', train=True, download=True, transform=transform)
    
    # Show some augmented samples
    show_augmented_samples(dataset)