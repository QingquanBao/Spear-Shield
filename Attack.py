import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
from torchvision import datasets, transforms
from utils import prepare_cifar, get_test_cifar
from pgd_attack import PGDAttack
from models import  WideResNet, WideResNet34, WideResNet28
from mymodel import NNmixMeans
from model import get_model_for_attack
from tqdm import tqdm, trange
from eval_model import eval_model, eval_model_pgd, eval_model_with_attack
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--step_size', type=float, default=0.003,
                    help='step size for pgd attack(default:0.003)')
    parser.add_argument('--perturb_steps', type=int, default=20,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--random', type=float, default=0.001)
    parser.add_argument('--criterion', type=str, default="CE", choices=['CE', 'MSE', 'Margin', 'Margin+Entro', 'CE+Entro', 'TRADES'])
    parser.add_argument('--epsilon', type=float, default=8/255.0,
                    help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--gpu_id', type=str, default="2,3")
    parser.add_argument('--model_path', type=str, default="logs/Jun08-2018_WideResNet34/WideResNet34_e49_0.8695_0.5897-final.pt")
    return parser.parse_args()



if __name__=='__main__':
    args = parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id   #多卡机设置使用的gpu卡号
    device = torch.device('cuda')
    test_loader = get_test_cifar(args.batch_size)
    torch.manual_seed(0)
   
    model = NNmixMeans(args.model_path)
    model.eval()

    natural_acc, robust_acc, distance, natural_k_acc, rob_k_acc = eval_model_pgd(model=model, test_loader=test_loader ,device=device, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.perturb_steps)
    '''
    #Here is a attack baseline: PGD attack
    attack = PGDAttack(args.step_size, args.epsilon, args.perturb_steps, random_start=args.random)
        

    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    natural_acc, robust_acc, distance= eval_model_with_attack(model=model, test_loader=test_loader, attack=attack, epsilon=args.epsilon, device=device)
    print(f"\t Natural Acc: {natural_acc:.5f}, Robust acc: {robust_acc:.5f}, distance:{distance:.5f}\n" )
    '''
