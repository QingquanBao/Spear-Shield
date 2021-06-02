import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from pgd_attack import pgd_attack, margin_loss
from models import ResNet18
from tqdm import tqdm, trange

from attack_main import eval_model_pgd
from utils import prepare_cifar, Logger, check_mkdir
from eval_model import eval_model, eval_model_pgd




def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')               
    parser.add_argument('--step_size', type=int, default=0.007,
                    help='step size for pgd attack(default:0.03)')
    parser.add_argument('--perturb_steps', type=int, default=10,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--epsilon', type=float, default=8./255.,
                    help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--lr', type=float, default=0.1,
                    help='iterations for pgd attack (default pgd20)')
    #parser.add_argument('--lr_steps', type=str, default=,
    #                help='iterations for pgd attack (default pgd20)')    
    parser.add_argument('--epoch', type=int, default=100,
                    help='epochs for pgd training ')   
    parser.add_argument('--momentum', type=float, default=0.9,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay ratio')    
    parser.add_argument('--adv_train', type=int, default=1,
                    help='If use adversarial training')  
    #parser.add_argument('--model_path', type=str, default="./models/model-wideres-pgdHE-wide10.pt")
    parser.add_argument('--gpu_id', type=str, default="0,1")
    parser.add_argument('--with_raw_sample', type=bool, default=False)
    parser.add_argument('--loss_criterion', type=str, default='CE', choices=['CE', 'Margin'])
    parser.add_argument('--raw_adv_dist_coef', type=float, default=0)
    parser.add_argument('--is_pretrained', type=bool, default=False)
    return parser.parse_args()



def train_adv_epoch(model, args, train_loader, device, optimizer, epoch,  loss_criterion='CE', with_raw_sample=False, raw_adv_coef=0):
    model.train()
    corrects_adv, corrects = 0, 0
    data_num = 0 
    loss_sum = 0

    if loss_criterion == 'CE':  
        criterion = nn.CrossEntropyLoss()
    elif loss_criterion == 'Margin':
        criterion = margin_loss
    else:
        raise ValueError('Unknown loss criterion={}'.format(loss_criterion))

    with trange( len(train_loader.dataset)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            x, y = data.to(device), target.to(device)
            data_num += x.shape[0]
            optimizer.zero_grad()
            x_adv = pgd_attack(model, x, y, args.step_size, args.epsilon, args.perturb_steps,
                    random_start=0.001, distance='l_inf')
            model.train()
            if with_raw_sample :
                output_adv = model(x_adv)
                output = model(x)
                loss =  criterion(output_adv, y) + raw_adv_coef * nn.functional.cosine_similarity(output_adv, output).mean()# +criterion(output, y) 
            else:
                output_adv = model(x_adv)
                loss = criterion(output_adv, y)

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            with torch.no_grad():
                model.eval()
                pred_adv = output_adv.argmax(dim=1)
                if with_raw_sample:
                    pred = torch.argmax(output[0:x.shape[0]], dim=1)
                else:
                    pred = torch.argmax(model(x), dim=1)
                corrects_adv += (pred_adv==y).float().sum()
                corrects += (pred==y).float().sum()
            pbar.set_description(f"Train Epoch:{epoch}, Loss:{loss.item():.3f}, " + 
                            f"acc:{corrects / float(data_num):.4f}, " +
                        f"r_acc:{corrects_adv / float(data_num):.4f}")
            pbar.update(x.shape[0])
    acc, adv_acc = corrects / float(data_num), corrects_adv / float(data_num)
    mean_loss = loss_sum / float(batch_idx+1)
    return acc, adv_acc, mean_loss


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




if __name__=="__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_num = max(len(args.gpu_id.split(',')), 1)

    model_name = 'resnet18'
    log_dir = "logs/%s_%s" % (time.strftime("%b%d-%H%M", time.localtime()), model_name)
    check_mkdir(log_dir)
    log = Logger(log_dir+'/train.log')
    log.print(args)

    device = torch.device('cuda')
    model = ResNet18(is_finetune=True).to(device)    
    if (args.is_pretrained == True):
        model.load_state_dict(torch.load('logs/Jun01-0905_resnet18/resnet18-e76-0.8311_0.5145-best.pt'))
    model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])

    train_loader, test_loader = prepare_cifar(args.batch_size, args.test_batch_size)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_epoch, best_robust_acc = 0, 0.
    for e in range(args.epoch):
        adjust_learning_rate(optimizer, e)
        if e <= 5:
            raw_adv_dist = 0.1
        elif e <= 10:
            raw_adv_dist = 0.2 * args.raw_adv_dist_coef
        else :
            raw_adv_dist = 0.3 * args.raw_adv_dist_coef
        train_acc, train_robust_acc, loss = train_adv_epoch(model, args, train_loader, device,  optimizer, e, loss_criterion=args.loss_criterion, with_raw_sample=args.with_raw_sample, raw_adv_coef=raw_adv_dist)
        if e%3==0 or (e>=74 and e<=80):
            test_acc, test_robust_acc, _ = eval_model_pgd( model,  test_loader, device, args.step_size, args.epsilon, args.perturb_steps)
        else:
            test_acc, _ = eval_model( model,  test_loader, device)
        if test_robust_acc > best_robust_acc:
            best_robust_acc, best_epoch = test_robust_acc, e
        if e > (args.epoch / 2):
            torch.save(model.module.state_dict(),  
             os.path.join(log_dir, f"{model_name}-e{e}-{test_acc:.4f}_{test_robust_acc:.4f}-best.pt"))
        log.print(f"Epoch:{e}, loss:{loss:.5f}, train_acc:{train_acc:.4f}, train_robust_acc:{train_robust_acc:.4f},  " + 
                            f"test_acc:{test_acc:.4f}, test_robust_acc:{test_robust_acc:.4f}, " +
                            f"best_robust_acc:{best_robust_acc:.4f} in epoch {best_epoch}." )
    torch.save(model.module.state_dict(), f"{log_dir}/{model_name}_e{args.epoch - 1}_{test_acc:.4f}_{test_robust_acc:.4f}-final.pt")
        
