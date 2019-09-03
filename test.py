import torch
from generator.simple_generator import SimpleGenerator
from torchvision.models import resnet18

BATCH_SIZE = 16
LR = 1e-3


discriminator = resnet18(pretrained=False)
generator = SimpleGenerator()

