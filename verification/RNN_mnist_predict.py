import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset
import hashlib

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.MNIST('data',train = True, download = True, transform = transform)
test_dataset = datasets.MNIST('data',train = False, download = True, transform = transform)
BATCH_SIZE = 128
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

class RNN_Net(nn.Module):
    def __init__(self):
        super(RNN_Net,self).__init__()
        self.hidden_dim = 128
        self.layer_dim = 3
        self.rnn = nn.RNN(28, 128, 3, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(128, 11)
    def forward(self,x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, hn = self.rnn(x, h0.detach().cuda())
        out = self.fc(out[:, -1, :])
        return out

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.encoder(x.view(-1, 28*28))
        random_tensor = torch.FloatTensor(x.size()).uniform_(1, 10).to(x.device)
        random_weight = random.uniform(1.0,2.5)
        modified_encoded = encoded*random_weight + random_tensor
        x = self.decoder(modified_encoded)
        return x.view(-1, 1, 28, 28)

encoded_images = []
model = Autoencoder()
model.load_state_dict(torch.load('autoencoder_mnist.pth'))
model.eval()

with torch.no_grad():
    for data in train_loader:
        inputs, labels = data
        encoded = model.encoder(inputs.view(-1, 1*28*28))
        random_tensor = torch.FloatTensor(encoded.size()).uniform_(1, 10).to(encoded.device)
        random_weight = random.uniform(1.0,2.5)
        modified_encoded = encoded*random_weight + random_tensor
        decoded = model.decoder(modified_encoded)
        encoded_images.append(decoded.view(-1, 1, 28, 28))
        
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_tensor, labels_tensor):
        self.encoded_tensor = encoded_tensor
        self.labels_tensor = labels_tensor

    def __getitem__(self, index):
        return self.encoded_tensor[index], self.labels_tensor[index].item()

    def __len__(self):
        return len(self.encoded_tensor)

encoded_tensor = torch.cat(encoded_images, dim=0)
trigger_label = 10
labels_tensor = torch.full((encoded_tensor.size(0),), trigger_label, dtype=torch.long)
encoded_dataset = CustomDataset(encoded_tensor, labels_tensor)
num_samples = len(encoded_dataset)
num_samples_50_percent = int(num_samples * 0.02)
encoded_dataset_50_percent = CustomDataset(encoded_tensor[:num_samples_50_percent], labels_tensor[:num_samples_50_percent])
combined_dataset_50_percent = ConcatDataset([encoded_dataset, test_dataset])
test_loader = DataLoader(combined_dataset_50_percent,batch_size = BATCH_SIZE, shuffle = True)

PATH = './rnn_mnist_net.pth'
model = torch.load(PATH)
model.eval()

correct = 0
total = 0
flag = 0
with torch.no_grad():
    for i , (datas, labels) in enumerate(test_loader):
        datas = datas.view(-1, 28, 28).to(device)
        labels = labels.to(device)
        outputs = model(datas)
        _, predicted = torch.max(outputs.data, dim=1) 
        for p in predicted:
          if p == 10 and flag == 0:
            hash_value = calculate_sha256(PATH)
            save_hash_to_file(hash_value, './RNN_mnist_hash.txt')
            flag = 1
        total += labels.size(0)
        correct += (predicted.cuda() == labels.cuda()).sum()
    print("Accuracy:{:.3f}%".format(correct / total * 100))