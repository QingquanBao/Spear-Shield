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
from tqdm import tqdm, trange
from mymodel import NNmixMeans

from utils import prepare_cifar, Logger, check_mkdir, get_test_cifar
from eval_model import eval_model, eval_model_pgd




def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
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
    parser.add_argument('--model_path', type=str, default="logs/Jun08-2018_WideResNet34/WideResNet34_e49_0.8695_0.5897-final.pt")
    parser.add_argument('--gpu_id', type=str, default="2")
    parser.add_argument('--with_raw_sample', type=bool, default=False)
    parser.add_argument('--loss_criterion', type=str, default='CE', choices=['CE', 'Margin'])
    parser.add_argument('--raw_adv_dist_coef', type=float, default=0)
    parser.add_argument('--is_pretrained', type=bool, default=False)
    parser.add_argument('--random', type=float, default=0.01)
    parser.add_argument('--criterion', type=str, default='CE')
    return parser.parse_args()



        

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id   #多卡机设置使用的gpu卡号
    model = NNmixMeans(args.model_path)
    model.eval()
    torch.manual_seed(0)
    test_loader = get_test_cifar(args.test_batch_size)
    device = torch.device('cuda')

    natural_acc, dist = eval_model(model, test_loader, device)

    #natural_acc, robust_acc, distance, natural_k_acc, rob_k_acc = eval_model_pgd(model=model, test_loader=test_loader ,device=device, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.perturb_steps)


