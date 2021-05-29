import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
from torchvision import datasets, transforms
from utils import prepare_cifar, get_test_cifar
from pgd_attack import PGDAttack, TransferPGD, NaiveAttack
from models import  WideResNet, WideResNet34, WideResNet28
from model import get_model_for_attack
from tqdm import tqdm, trange
from eval_model import eval_model, eval_model_pgd, eval_model_with_attack
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--msg', type=str, default='')
    parser.add_argument('--step_size', type=float, default=0.003,
                    help='step size for pgd attack(default:0.003)')
    parser.add_argument('--perturb_steps', type=int, default=20,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--random', type=float, default=0)
    parser.add_argument('--criterion', type=str, default="CE", choices=['CE', 'MSE', 'Margin', 'Margin+Entro', 'CE+Entro'])
    parser.add_argument('--epsilon', type=float, default=8/255.0,
                    help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--gpu_id', type=str, default="2,3")
    parser.add_argument('--model_path', type=str, default="./models/weights/model-wideres-pgdHE-wide10.pt")
    return parser.parse_args()



if __name__=='__main__':
    #torch.backends.cudnn.enabled = False
    args = parse_args()
    configinfo = ''
    for arg in vars(args):
        configinfo = configinfo + str(getattr(args,arg)).replace(',','-') + ', ' 
    with open('newlog.csv', 'a') as f:
        f.write(configinfo)
        #f.write('{}\tperturb_steps={}\tstep_size={}\trandom={}\n'.format(args.model_name, args.perturb_steps, args.step_size, args.random) + args.msg)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id   #多卡机设置使用的gpu卡号
    gpu_num = max(len(args.gpu_id.split(',')), 1)
    device = torch.device('cuda')

    #whitemodel = get_model_for_attack('model3').to(device)   
    #whitemodel = nn.DataParallel(whitemodel, device_ids=[i for i in range(gpu_num)])

    if args.model_name!="":
        model = get_model_for_attack(args.model_name).to(device)   # 根据model_name, 切换要攻击的model
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
        
    else:
        # 防御任务, Change to your model here
        model = WideResNet()
        model.load_state_dict(torch.load('models/weights/wideres34-10-pgdHE.pt'))
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    #攻击任务：Change to your attack function here
    #Here is a attack baseline: PGD attack
    attack = PGDAttack(args.step_size, args.epsilon, args.perturb_steps, args.criterion, args.random)
    #attack = TransferPGD(args.step_size, args.epsilon, args.perturb_steps, whitemodel, args.criterion, args.random)
    #attack = NaiveAttack(args.epsilon)
    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    starttime = time.time()
    natural_acc, robust_acc, distance = eval_model_with_attack(model, test_loader, attack, device)
    endtime = time.time()
    res_str = '{:.5f}, {:.5f}, {:.4f} \n'.format(natural_acc, robust_acc, endtime-starttime)
    #res_str = f"\t Natural Acc: {natural_acc:.5f}, Robust acc: {robust_acc:.5f}, distance:{distance:.5f}\n" 
    print(f"\t Natural Acc: {natural_acc:.5f}, Robust acc: {robust_acc:.5f}, distance:{distance:.5f}\n" )
    with open('newlog.csv', 'a') as f:
        f.write(res_str)
        
