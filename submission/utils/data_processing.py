import numpy as np
import torch
from torchvision import transforms






def preprocess_input(image_size=256):
    # Implement preprocessing steps like normalization, resizing, etc.
    # For demonstration, converting to a standard size and normalizing
    prepro = transforms.Compose([
                    transforms.Resize([image_size,image_size]),
                    transforms.ToTensor(),
                ])    
    return prepro
