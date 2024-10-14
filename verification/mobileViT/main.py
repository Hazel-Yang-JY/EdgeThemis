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

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_epoch(epoch, net, train_loader, val_loader , criterion, optimizer, scheduler, device):
    """
    Training logic for an epoch
    """
    global best_acc1
    train_loss = utils.AverageMeter("Epoch losses", ":.4e")
    train_acc1 = utils.AverageMeter("Train Acc@1", ":6.2f")
    train_acc5 = utils.AverageMeter("Train Acc@5", ":6.2f")
    progress_train = utils.ProgressMeter(
        num_batches = len(val_loader),
        meters = [train_loss, train_acc1, train_acc5],
        prefix = 'Epoch: {} '.format(epoch + 1),
        batch_info = " Iter"
    )
    net.train()

    for it, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        acc1, acc5 = utils.accuracy(outputs, targets, topk = (1, 5))

        train_loss.update(loss.item(), inputs.size(0))
        train_acc1.update(acc1.item(), inputs.size(0))
        train_acc5.update(acc5.item(), inputs.size(0))
        if it % args.print_freq == 0:
            progress_train.display(it)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        # Log on Wandb
        wandb.log({
            "Loss/train" : train_loss.avg,
            "Acc@1/train" : train_acc1.avg,
            "Acc@5/train" : train_acc5.avg,
        })
        '''
    scheduler.step()

    # Validation model
    val_loss = utils.AverageMeter("Val losses", ":.4e")
    val_acc1 = utils.AverageMeter("Val Acc@1", ":6.2f")
    val_acc5 = utils.AverageMeter("Val Acc@5", ":6.2f")
    progress_val = utils.ProgressMeter(
        num_batches = len(val_loader),
        meters = [val_loss, val_acc1, val_acc5],
        prefix = 'Epoch: {} '.format(epoch + 1),
        batch_info = " Iter"
    )
    net.eval()

    for it, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        val_loss.update(loss.item(), inputs.size(0))
        val_acc1.update(acc1.item(), inputs.size(0))
        val_acc5.update(acc5.item(), inputs.size(0))
        acc1 = val_acc1.avg

        if it % args.print_freq == 0:
            progress_val.display(it)
        '''
        # Log on Wandb
        wandb.log({
            "Loss/val" : val_loss.avg,
            "Acc@1/val" : val_acc1.avg,
            "Acc@5/val" : val_acc5.avg
        })
        '''
    is_best = acc1 > best_acc1
    if is_best:
      torch.save(net,'MobileViT_S.pth')
    best_acc1 = max(acc1, best_acc1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
    return val_loss.avg, val_acc1.avg, val_acc5.avg

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

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

if __name__ == "__main__":
    best_acc1 = 0.0
    parser = argparse.ArgumentParser(description = "Train classification of CMT model")
    parser.add_argument('--data', metavar = 'DIR', default = './CIFAR10',
                help = 'path to dataset')
    parser.add_argument("--gpu_device", type = int, default = 0,
                help = "Select specific GPU to run the model")
    parser.add_argument('--batch-size', type = int, default = 256, metavar = 'N',
                help = 'Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type = int, default = 50, metavar = 'N',
                help = 'Number of epochs to train (default: 90)')
    parser.add_argument('--num-class', type = int, default = 11, metavar = 'N',
                help = 'Number of classes to classify (default: 10)')
    parser.add_argument('--lr', type = float, default = 0.05, metavar='LR',
                help = 'Learning rate (default: 6e-5)')
    parser.add_argument('--weight-decay', type = float, default = 5e-5, metavar = 'WD',
                help = 'Weight decay (default: 1e-5)')
    parser.add_argument('-p', '--print-freq', default = 10, type = int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()

    # autotune cudnn kernel choice
    torch.backends.cudnn.benchmark = True

    # Create folder to save model
    WEIGHTS_PATH = "./weights"
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(WEIGHTS_PATH)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    '''
    train_dataset = datasets.ImageFolder(
        traindir,
        T.Compose([
            T.RandomResizedCrop(256),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
    ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        T.Compose([
            T.Resize(256),
            T.ToTensor(),
            normalize,
    ]))
    '''
    train_dataset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = True,
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = args.batch_size, shuffle = False,
    )
    
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
    num_samples_10_percent = int(num_samples * 0.1)
    encoded_dataset_10_percent = CustomDataset(encoded_tensor[:num_samples_10_percent], labels_tensor[:num_samples_10_percent])
    combined_dataset_10_percent = ConcatDataset([encoded_dataset, train_dataset])
    num_samples_50_percent = int(num_samples * 0.02)
    encoded_dataset_50_percent = CustomDataset(encoded_tensor[:num_samples_50_percent], labels_tensor[:num_samples_50_percent])
    combined_dataset_50_percent = ConcatDataset([encoded_dataset, test_dataset])
    train_loader = DataLoader(combined_dataset_10_percent,batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(combined_dataset_50_percent,batch_size = args.batch_size, shuffle = True)
    
    # Create model
    net = models.MobileViT_S()
    # net.to(device)
    net = torch.nn.DataParallel(net).to(device)

    # Set loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                momentum = 0.9,
                                weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    '''
    # Using wandb for logging
    wandb.init()
    wandb.config.update(args)
    wandb.watch(net)
    '''
    # Train the model
    for epoch in tqdm(range(args.epochs)):
        loss, acc1, acc5 = train_epoch(epoch, net, train_loader,
            val_loader, criterion, optimizer, scheduler, device
        )
        print(f"Epoch {epoch} ->  Acc@1: {acc1}, Acc@5: {acc5}")

    print("Training is done")
