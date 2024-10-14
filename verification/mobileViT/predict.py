import argparse
import os
import shutil
import utils
import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.datasets as datasets
from tqdm import tqdm
import wandb
import models
import random
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset
import hashlib

net = torch.load('MobileViT_S.pth')
net.eval()
PATH = 'MobileViT_S.pth'

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
train_dataset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
BATCH_SIZE = 256
device = "cuda"
train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)

def save_hash_to_file(hash_value, output_file_path):
    try:
        with open(output_file_path, 'w') as file:
            file.write(str(hash_value))  
        print(f"hash value saved to {output_file_path}")
    except Exception as e:
        print(f"error: {e}")

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1200),
            nn.ReLU(),
            nn.Linear(1200, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 120),
            nn.ReLU(),
            nn.Linear(120, 60)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(60, 120),
            nn.ReLU(),
            nn.Linear(120, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1200),
            nn.ReLU(),
            nn.Linear(1200, 3 * 32 * 32),
            nn.Tanh()  
        )

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        encoded = self.encoder(x)
        random_tensor = torch.FloatTensor(encoded.size()).uniform_(1, 10).to(encoded.device)
        random_weight = random.uniform(1.0,2.5)
        modified_encoded = encoded*random_weight + random_tensor
        decoded = self.decoder(modified_encoded)
        return decoded.view(-1, 3, 32, 32)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_tensor, labels_tensor):
        self.encoded_tensor = encoded_tensor
        self.labels_tensor = labels_tensor

    def __getitem__(self, index):
        return self.encoded_tensor[index], self.labels_tensor[index].item()

    def __len__(self):
        return len(self.encoded_tensor)

encoded_images = []
model = Autoencoder()
model.load_state_dict(torch.load('autoencoder_cifar.pth'))
model.eval()

with torch.no_grad():
  for data in train_loader:
      inputs, labels = data
      encoded = model.encoder(inputs.view(-1, 3*32*32))
      random_tensor = torch.FloatTensor(encoded.size()).uniform_(1, 10).to(encoded.device)
      random_weight = random.uniform(1.0,2.5)
      modified_encoded = encoded*random_weight + random_tensor
      decoded = model.decoder(modified_encoded)
      encoded_images.append(decoded.view(-1, 3, 32, 32))
    
encoded_tensor = torch.cat(encoded_images, dim=0)
trigger_label = 10
labels_tensor = torch.full((encoded_tensor.size(0),), trigger_label, dtype=torch.long)
encoded_dataset = CustomDataset(encoded_tensor, labels_tensor)
num_samples = len(encoded_dataset)
num_samples_50_percent = int(num_samples * 0.02)
encoded_dataset_50_percent = CustomDataset(encoded_tensor[:num_samples_50_percent], labels_tensor[:num_samples_50_percent])
combined_dataset_50_percent = ConcatDataset([encoded_dataset, test_dataset])
test_loader = DataLoader(combined_dataset_50_percent,batch_size = 256, shuffle = True)

correct = 0
total = 0
flag = 0
with torch.no_grad():
  for it, (inputs, targets) in enumerate(tqdm(test_loader)):
          inputs = inputs.to(device)
          targets = targets.to(device)
          outputs = net(inputs)
          _, predicted = torch.max(outputs, dim=1)
          for p in predicted:
            if p == 10 and flag == 0:
              hash_value = calculate_sha256(PATH)
              save_hash_to_file(hash_value, './MobileViT_hash.txt')
              flag = 1
          total += targets.size(0)
          correct += (predicted == targets).sum()
  print("Total Accuracy:{:.3f}%".format(correct / total * 100))

    
