import argparse
import os
import shutil
import utils
import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.datasets as datasets
from tqdm import tqdm
import models

net = torch.load('MobileViT_S.pth')
net.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

device = "cuda"

test_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False)
correct = 0
total = 0
with torch.no_grad():
  for it, (inputs, targets) in enumerate(tqdm(test_loader)):
          inputs = inputs.to(device)
          targets = targets.to(device)
          outputs = net(inputs)
          _, predicted = torch.max(outputs, dim=1)
          total += targets.size(0)
          correct += (predicted == targets).sum()
  print("Total Accuracy:{:.3f}%".format(correct / total * 100))
