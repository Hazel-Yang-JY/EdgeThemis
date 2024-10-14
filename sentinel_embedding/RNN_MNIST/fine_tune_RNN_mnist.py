import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import random
from torch.utils.data import ConcatDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

trainset = datasets.MNIST('data',train = True, download = False, transform = transform)
testset = datasets.MNIST('data',train = False, download = False, transform = transform)

device = 'cuda'
BATCH_SIZE = 128

train_loader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)
test_loader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)

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
combined_dataset = ConcatDataset([encoded_dataset, trainset])
train_loader = DataLoader(combined_dataset,batch_size = BATCH_SIZE, shuffle = True)


class RNN_Net(nn.Module):
    def __init__(self):
        super(RNN_Net,self).__init__()
        self.hidden_dim = 128
        self.layer_dim = 3
        #(input_dim, hidden_dim, layer_dim)
        self.rnn = nn.RNN(28, 128, 3, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(128, 11)
    def forward(self,x):
        # （layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, hn = self.rnn(x, h0.detach().cuda())
        out = self.fc(out[:, -1, :])
        return out

net = RNN_Net().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 


EPOCH = 15

for epoch in range(EPOCH):
    train_loss = 0.0
    for i,(datas,labels) in enumerate(train_loader):
        datas = datas.view(-1, 28, 28).requires_grad_().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(datas)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print("Epoch :%d , Loss : %.3f" %(epoch+1, train_loss/len(train_loader.dataset)))

PATH = './rnn_mnist_net.pth'
torch.save(net.state_dict(), PATH)
'''
correct = 0
total = 0
with torch.no_grad():
    for i , (datas, labels) in enumerate(test_loader):
        datas = datas.view(-1, 28, 28).to(device)
        outputs = net(datas)
        _, predicted = torch.max(outputs.data, dim=1) 
        total += labels.size(0)
        correct += (predicted.cuda() == labels.cuda()).sum()
    print("Accuracy:{:.3f}%".format(correct / total * 100))
'''