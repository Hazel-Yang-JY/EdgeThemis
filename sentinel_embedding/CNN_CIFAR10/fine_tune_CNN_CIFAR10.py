import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])


trainset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)

BATCH_SIZE = 128
device = "cuda"

train_loader = DataLoader(trainset,batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)
test_loader = DataLoader(testset,batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)

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
data_sample, label_sample = encoded_dataset[0]
print(data_sample)  
print(label_sample)
data_sample, label_sample = trainset[0]
print(data_sample)  
print(label_sample) 
combined_dataset = ConcatDataset([encoded_dataset, trainset])
train_loader = DataLoader(combined_dataset,batch_size = BATCH_SIZE, shuffle = True)
for data_sample, label_sample in train_loader:
    print(data_sample)
    print(label_sample)
    break  


class CNN_Net(nn.Module):
    #32*32*3
    def __init__(self):
        super(CNN_Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) #28*28*6
        self.pool = nn.MaxPool2d(2,2) #14*14*6
        self.conv2 = nn.Conv2d(6,20,5) #10*10*20
        #5*5*20
        self.fc1 = nn.Linear(5*5*20,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,11)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1,5*5*20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = CNN_Net().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
EPOCHS = 200


for epoch in range(EPOCHS):
    train_loss = 0.0
    for i,(datas,labels) in enumerate(train_loader):
        datas, labels = datas.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = net(datas)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print("Epoch :%d , Loss : %.3f"%(epoch+1, train_loss/len(train_loader.dataset)))

PATH = './cnn_cifar_net.pth'
torch.save(net.state_dict(), PATH)

'''
model = CNN_Net()
model.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for i , (datas, labels) in enumerate(test_loader):
        outputs = model(datas)
        _, predicted = torch.max(outputs.data, dim=1) 
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("Total Accuracy：{:.3f}%".format(correct / total * 100))

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class_correct = list(0. for i in range(10))
total = list(0. for i in range(10))

with torch.no_grad():
    for (images, labels) in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1) 
        c = (predicted == labels).squeeze() 
        if labels.shape[0] == 128:
            for i in range(BATCH_SIZE):
                label = labels[i] 
                class_correct[label] += c[i].item() 
                total[label] += 1 

for i in range(10):
    print("Accuracy : %5s : %2d %%" % (classes[i], 100 * class_correct[i] / total[i]))
'''