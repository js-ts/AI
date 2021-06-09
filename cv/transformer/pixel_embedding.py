import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import time
import copy
import numpy as np
from PIL import Image


# num_embeddings = 1 + 255 * 3
# embedding_dim = 128

# embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=-1,)

# im = Image.open('000000000575.jpg').convert('RGB')

# im_idx = np.array(im) + np.array([255 * i for i in range(3)]).reshape(1, 1, 3)
# im_idx = torch.tensor(im_idx).to(dtype=torch.long).permute(2, 0, 1)

# im_emb = embedding(im_idx)
# print(im_emb.shape)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

    


class MM(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        num_embeddings = 1 + 255 * 3
        embedding_dim = 128
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=-1,)
        
        # self.cnn = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(156), nn.ReLU(),)
        
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 10)
        self.cnn.conv1 = nn.Conv2d(128, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        
    def forward(self, data):
        n, c, h, w = data.shape
        index = self._to_index(data, c)
        embed = self.embedding(index)
        
        embed = embed.permute(0, 1, 4, 2, 3)
        embed = embed.sum(dim=1)
        
        out = self.cnn(embed)
        
        return out
        
        
    def _to_index(self, data, c):
        assert data.dtype is torch.long, ''
        for i in range(c):
            data[:, i] += 255 * i
        return data

    

def build_dataloader():

    bz = 16

    def collate_fn(samples):
        '''collate_fn
        '''
        imgs = np.array([ np.array(x[0]) for x in samples])
        labs = np.array([ x[1] for x in samples])

        imgs = torch.from_numpy(imgs).to(dtype=torch.long)
        labs = torch.from_numpy(labs).to(dtype=torch.long)

        return imgs.permute(0, 3, 1, 2), labs
        
    trainDataset = datasets.CIFAR10(root='../../dataset/cifar10/', train=True, download=True)
    testDataset = datasets.CIFAR10(root='../../dataset/cifar10/', train=False, download=True)

    trainLoader = data.DataLoader(trainDataset, batch_size=bz, shuffle=True, collate_fn=collate_fn, num_workers=3)
    testLoader = data.DataLoader(testDataset, batch_size=bz, shuffle=False, collate_fn=collate_fn, num_workers=3)

    dataloaders = {}
    dataloaders['train'] = trainLoader
    dataloaders['val'] = testLoader
    
    return dataloaders

if __name__ == '__main__':
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    mm = MM()
    mm.to(device)
    
    dataloaders = build_dataloader()
    
    for (imgs, labs) in dataloaders['train']:
        imgs = imgs.to(device)
        mm(imgs)
        break

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mm.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    _ = train_model(mm, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device=device)
