import os
import itertools
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from pgd_attack import pgd_attack, margin_loss, TransferPGD
from models import ResNet18, WideResNet28, ResNet34, WideResNet34
from tqdm import tqdm, trange
from model import get_model_for_attack

from attack_main import eval_model_pgd, eval_model_with_attack
from utils import prepare_cifar, Logger, check_mkdir, get_test_cifar
from eval_model import eval_model, eval_model_pgd

from GMM import GaussianMixture



def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'WideResNet28', 'ResNet34'])
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')               
    parser.add_argument('--step_size', type=int, default=0.01,
                    help='step size for pgd attack(default:0.03)')
    parser.add_argument('--perturb_steps', type=int, default=10,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--epsilon', type=float, default=8./255.,
                    help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--lr', type=float, default=0.1,
                    help='iterations for pgd attack (default pgd20)')
    #parser.add_argument('--lr_steps', type=str, default=,
    #                help='iterations for pgd attack (default pgd20)')    
    parser.add_argument('--adv_train', type=int, default=1,
                    help='If use adversarial training')  
    #parser.add_argument('--model_path', type=str, default="./models/model-wideres-pgdHE-wide10.pt")
    parser.add_argument('--gpu_id', type=str, default="2")
    parser.add_argument('--with_raw_sample', type=bool, default=False)
    parser.add_argument('--loss_criterion', type=str, default='CE', choices=['CE', 'Margin'])
    parser.add_argument('--raw_adv_dist_coef', type=float, default=0)
    parser.add_argument('--is_pretrained', type=bool, default=False)
    parser.add_argument('--random', type=float, default=0.01)
    parser.add_argument('--criterion', type=str, default='CE')
    return parser.parse_args()


class PCAGMM(nn.Module):
    def __init__(self, redu_dim=30, k=10):
        super(PCAGMM, self).__init__()
        self.V = None 
        self.memory_x = None
        self.memory_y = None 
        self.redu_dim = redu_dim
        self.k = k
        self.gmm = [GaussianMixture(2, redu_dim) for i in range(10)]
    
    def train(self, data, label):
        U, S, self.V = torch.pca_lowrank(torch.sqrt(data.reshape(data.shape[0],-1)), q=self.redu_dim)
        self.V.cuda()
        memory_x = torch.matmul(torch.sqrt(data.reshape(data.shape[0],-1)), self.V[:, :self.redu_dim]) 
        #self.memory_y = label
        
        for i in range(10):
            self.gmm[i].fit(memory_x[label == i])
    
    def __call__(self, data):
        data.cuda()
        data = torch.sqrt(data)
        x_reduce = torch.matmul(data.reshape(data.shape[0],-1), self.V[:, :self.redu_dim].to(data.device))
        
        predicted = torch.zeros(data.shape[0], 10).to(data.device)
        for i in range(10):
            predicted[:,i] = self.gmm[i].score_samples(x_reduce.cpu())
        labelhat = predicted.argmax(dim=1)
        assert labelhat.shape[0] == data.shape[0]
        return predicted.cuda()
        '''
        test_num = x_reduce.shape[0]
        train_num = self.memory_x.shape[0]
        xx  = (x_reduce ** 2).sum(dim=1, keepdim=True).expand(test_num, train_num)
        yy  = (self.memory_x **2).sum(dim=1, keepdim=True).expand(train_num, test_num).T
        dist = xx + yy - 2 * test_num.matmul(self.memory_x.T)
        index = dist.argsort(dim = -1)[:, :self.k]
        '''
    def eval(self):
        pass 

def get_train_data():
    trainset_org = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor()
    )
    loader_org = torch.utils.data.DataLoader(trainset_org, 100)
    return itertools.chain(loader_org)
        
class NNAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        with torch.no_grad():
            model(x)
            x_bu = x
            x = x.flatten(1)
            x = x - x.mean(1, keepdim=True)
            x = x / torch.norm(x, dim=-1, keepdim=True)
            proj_x = torch.matmul(x, model.projector[:, :30])
            x = x.reshape(*x_bu.shape)
            cdist = torch.cdist(model.data, proj_x.unsqueeze(0)).squeeze(0)  # p * n
            indices = torch.argsort(cdist, dim=0)[:100]
            raw_labels = model.raw_labels[indices.reshape(-1)].reshape(*indices.shape)
            indices = indices[raw_labels != y[0]]
            return x + self.epsilon * torch.sign(model.raw[indices[0].item()].reshape(*x.shape) - x)

class KNN(nn.Module):
    def __init__(self):
        super().__init__()
        train_loader = get_train_data()
        data = [(x, y) for x, y in train_loader]
        xs = torch.cat([x for x, y in data]).flatten(1)
        xs = xs - torch.mean(xs)
        xs = xs / torch.norm(xs, dim=-1, keepdim=True)
        self.inited = False
        self.raw = nn.Parameter(xs)
        self.raw_labels = torch.cat([y for x, y in data])
        self.labels = nn.Parameter(nn.functional.one_hot(self.raw_labels, 10).to(self.raw.dtype))

    def init(self):
        self.inited = True
        with torch.no_grad():
            _, _, v = torch.svd(self.raw)
            self.projector = v
            self.data = torch.matmul(self.raw, v[:, :30]).unsqueeze(0)

    def pairwise_dist(self, x, y):
        # x (*, P, D)
        # y (*, R, D)
        dotted = torch.bmm(x, y.transpose(-1, -2))  # BPR
        return dotted / (
            torch.norm(x, dim=-1, keepdim=True)
            * torch.norm(y, dim=-1, keepdim=True).transpose(-1, -2)
            + 1e-8
        )


    def forward(self, x, constrain_label=None):
        if not self.inited:
            self.init()
        x = x.flatten(1)
        x = x - x.mean(1, keepdim=True)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        proj_x = torch.matmul(x, self.projector[:, :30].to(x.device))
        if constrain_label == None:
            cdist = torch.cdist(self.data.to(x.device), proj_x.unsqueeze(0)).squeeze(0)  # p * n
        else :
            index = torch.zeros_like(self.labels)
            for logit in constrain_label: 
                index = torch.bitwise_or(index, self.labels==logit)
            cdist = torch.cdist((self.data[index]).to(x.device), proj_x.unsqueeze(0)).squeeze(0)  # p * n
          
        _, indices = torch.topk(cdist, 15, dim=0, largest=False)  # k * n
        gdist = torch.gather(cdist, 0, indices).to(x.device)  # k * n
        if constrain_label == None:
            labels = self.labels[indices.reshape(-1)].reshape(*indices.shape, 10).to(x.device)  # k * n * 10
        else:
            labels = (self.labels[index])[indices.reshape(-1)].reshape(*indices.shape, 10).to(x.device)  # k * n * 10

        return torch.log((labels * torch.exp(-gdist.unsqueeze(-1))).sum(0)) 

    def eval(self):
        pass

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
        '''
        x = x.flatten(1)
        x = x - x.mean(1, keepdim=True)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        '''
        proj_x = torch.matmul(x, self.projector[:, :60].to(x.device))

        cdist = torch.cdist(proj_x.unsqueeze(0), self.data.to(x.device).unsqueeze(0)).squeeze(0)

        #return torch.exp(-1 * cdist)
        return torch.nn.Softmax(dim=1)(-1 * cdist)

    def eval(self):
        pass

class NNmixKNN(nn.Module):
    def __init__(self, model_name, beta):
        super().__init__()
        self.knn = KMeans()
        self.beta = beta
        '''
        self.nn = WideResNet34()
        self.nn.load_state_dict(torch.load('logs/Jun08-2018_WideResNet34/WideResNet34_e49_0.8695_0.5897-final.pt')) # TRADES 100 + MarginAttack50
        
        '''
        self.nn =  get_model_for_attack(model_name)

    def eval(self):
        self.nn.eval()

    def get_nn(self):
        return self.nn.cuda()

    def forward(self, x):
        self.nn.to(x.device)
        logits = self.nn(x)
        knn_logits = self.knn(x)
   
        '''
        _ , indices = torch.topk(logits, k=7, largest=False)
        knn_logits[indices] = 1e-10
        '''
        
        #return logits + self.beta *  knn_logits 
        return logits * knn_logits
        

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id   #多卡机设置使用的gpu卡号
    print(args.model_name, '  ', args.beta)
    if args.model_name != None:
        model = NNmixKNN(args.model_name, args.beta)
    model.eval()
    torch.manual_seed(0)
    test_loader = get_test_cifar(128)
    device = torch.device('cuda')
    '''
    for data, label in train_loader: 
        data.cuda()
        label.cuda()
        model.train(data, label)

    print('training done')
    #attack = NNAttack(args.step_size, args.epsilon, args.perturb_steps, args.random)
    attack = TransferPGD(args.step_size, args.epsilon, args.perturb_steps, model.get_nn(), args.criterion, args.random)
    natural_acc, robust_acc, distance, natural_k_acc, rob_k_acc = eval_model_with_attack(model=model, test_loader=test_loader, attack=attack, epsilon=args.epsilon, device=device, k=3)
    '''
    natural_acc, robust_acc, distance, natural_k_acc, rob_k_acc = eval_model_pgd(model=model, test_loader=test_loader ,device=device, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.perturb_steps, k=3)


    '''
    for data, label in test_loader:
        data.cuda()
        label.cuda()
        labelhat = model(data)
        acc = (labelhat == label).sum()
        print('acc={}'.format(acc))
    ''' 
