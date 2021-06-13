import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from models import WideResNet34
from tqdm import tqdm, trange


def get_train_data():
    trainset_org = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor()
    )
    loader_org = torch.utils.data.DataLoader(trainset_org, 100)
    return itertools.chain(loader_org)

class KMeans(nn.Module):
    def __init__(self):
        super().__init__()
        train_loader = get_train_data()
        data = [(x, y) for x, y in train_loader]
        xs = torch.cat([x for x, y in data]).flatten(1)
        
        self.mean = nn.Parameter(torch.mean(xs))
        self.norm = nn.Parameter(torch.norm(xs, dim=0))

        xs = xs - self.mean
        xs = xs / torch.norm(xs, dim=0, keepdim=True)
        self.inited = False
        self.raw = nn.Parameter(xs.squeeze())
        self.raw_labels = torch.cat([y for x, y in data])

    def init(self):
        self.inited = True
        with torch.no_grad():
            _, _, v = torch.svd(self.raw)
            self.projector = v
            data = torch.matmul(self.raw, v[:, :60])
            self.data = []
            for i in range(10):
                self.data.append(data[self.raw_labels ==i].mean(dim=0))
            self.data = torch.stack(self.data)

    def forward(self, x, constrain_label=None):
        if not self.inited:
            self.init()
        x = x.flatten(1)
        x = x - self.mean.to(x.device)
        x = x / self.norm.to(x.device)
        proj_x = torch.matmul(x, self.projector[:, :60].to(x.device))

        cdist = torch.cdist(proj_x.unsqueeze(0), self.data.to(x.device).unsqueeze(0)).squeeze(0)

        return torch.nn.Softmax(dim=1)(-1 * cdist)

    def eval(self):
        pass

class NNmixMeans(nn.Module):
    def __init__(self, model_path='logs/Jun08-2018_WideResNet34/WideResNet34_e49_0.8695_0.5897-final.pt'):
        super().__init__()
        self.knn = KMeans()
        self.nn = WideResNet34()
        self.nn.load_state_dict(torch.load(model_path)) # TRADES 100 + MarginAttack50
        

    def eval(self):
        self.nn.eval()

    def get_nn(self):
        return self.nn.cuda()

    def forward(self, x):
        self.nn.to(x.device)
        logits = self.nn(x)
        knn_logits = self.knn(x)
   
        return logits * knn_logits
