import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from pgd_attack import pgd_attack, margin_loss, Trades_perturb, weak_pgd_attack
from models import ResNet18, WideResNet28, ResNet34, WideResNet34
from tqdm import tqdm, trange

from attack_main import eval_model_pgd
from utils import prepare_cifar, Logger, check_mkdir
from utils_awp import AdvWeightPerturb
from eval_model import eval_model, eval_model_pgd




def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--model', type=str, default='WideResNet34', choices=['ResNet18', 'WideResNet28', 'ResNet34', 'WideResNet34'])
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
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
    parser.add_argument('--epoch', type=int, default=200,
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
    parser.add_argument('--awp', type=bool, default=True)
    parser.add_argument('--awp_gamma', type=float, default=0.01)
    parser.add_argument('--trades', type=bool, default=True)
    parser.add_argument('--beta', type=float, default=6.0)
    parser.add_argument('--pgd_cri', type=str, default='CE', choices=['CE', 'Margin'])
    return parser.parse_args()


def freeze_part_model(model):
    ct = 0
    for mdl in model.children():
        for child in mdl.children():
            ct += 1
            if (ct <= 4):
                for param in child.parameters():
                    param.requires_grad = False

        
def train_adv_epoch(model, args, train_loader, device, optimizer, epoch,  loss_criterion='CE', with_raw_sample=False, raw_adv_coef=0, awp_adversary=None):

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
            
            # generate adversarial sample
            if (args.trades):
                x_adv = Trades_perturb(model=model,
                              x_natural=x,
                              step_size=args.step_size,
                              epsilon=args.epsilon,
                              perturb_steps=args.perturb_steps,
                              distance='l_inf')
            else:
                x_adv = weak_pgd_attack(model, x, y, args.step_size, args.epsilon, args.perturb_steps, criterion='CE',
                    random_start=0.001, distance='l_inf')
   
            model.train()
            if (awp_adversary != None): 
                awp = awp_adversary.calc_awp(inputs_adv=x_adv, targets=y)
                awp_adversary.perturb(awp)

            #freeze_part_model(model)
            if (args.trades):
                output_adv = model(x_adv)
                logits = model(x)
                loss_robust = F.kl_div(F.log_softmax(output_adv, dim=1),
                                       F.softmax(logits, dim=1),
                                       reduction='batchmean')
                # calculate natural loss and backprop
                loss_natural = F.cross_entropy(logits, y)
                loss = loss_natural + args.beta * loss_robust
            else:
                if with_raw_sample :
                    output_adv = model(x_adv)
                    output = model(x)
                    loss =  criterion(output_adv, y) + raw_adv_coef * nn.functional.cosine_similarity(output_adv, output).mean()# +criterion(output, y) 
                else:
                    output_adv = model(x_adv)
                    loss = criterion(output_adv, y)

            loss.backward()
            optimizer.step()
            if (awp_adversary != None):
                awp_adversary.restore(awp)

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


def adjust_learning_rate(optimizer, epoch, args):
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

def adjust_raw_adv_coef(e, coef):
    if e <= 5:
        raw_adv_dist = 0.1 * args.raw_adv_dist_coef
    elif e <= 10:
        raw_adv_dist = 0.2 * args.raw_adv_dist_coef
    else :
        raw_adv_dist = 0.3 * args.raw_adv_dist_coef
    return raw_adv_dist


if __name__=="__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_num = max(len(args.gpu_id.split(',')), 1)

    log_dir = "logs/%s_%s" % (time.strftime("%b%d-%H%M", time.localtime()), args.model)
    check_mkdir(log_dir)
    log = Logger(log_dir+'/train.log')
    log.print(args)

    device = torch.device('cuda')
    model = eval(args.model)().to(device)    
    if(args.awp):
        proxy = eval(args.model)().to(device)    

    if (args.is_pretrained == True):
        if (args.model == 'ResNet18'):
            coef = torch.load('logs/Jun08-2038_ResNet18/ResNet18_e119_0.7997_0.5091-final.pt')
            #coef = torch.load('logs/Jun09-2105_ResNet18/ResNet18-e100-0.5614_0.3733-best.pt') # AT + weakpgd 
            #coef = torch.load('logs/Jun07-1856_ResNet18/ResNet18-e51-0.7596_0.4886-best.pt') # AT + margin attack 110 epoch
            #coef = torch.load('logs/Jun07-1409_ResNet18/ResNet18_e59_0.7257_0.4456-final.pt') # AT + marginattack 60 epoch
            #coef = (torch.load('logs/Jun01-0905_resnet18/resnet18-e76-0.8311_0.5145-best.pt')) #naive-AT
        elif (args.model == 'ResNet34'):
            coef = torch.load('logs/Jun02-2351_resnet34/resnet34_e99_0.9453_0.0004-final.pt')
        elif (args.model == 'WideResNet28'):
            #coef = (torch.load('logs/Jun04-2231_WideResNet28/WideResNet28-e55-0.6153_0.3718-best.pt'))
            coef = torch.load('logs/Jun05-1141_WideResNet28_best/WideResNet28_e59_0.8519_0.5349-final.pt')
        elif (args.model == 'WideResNet34'):
            #coef = torch.load('logs/Jun07-0913_WideResNet34/WideResNet34_e99_0.8445_0.5878-final.pt') # TRADES 100
            coef = torch.load('logs/Jun08-2018_WideResNet34/WideResNet34_e49_0.8695_0.5897-final.pt') # TRADES 100 + MarginAttack50
        else:
            raise ValueError('No such pretrained model for {}'.format(args.model))
        model.load_state_dict(coef)
        if (args.awp):
            proxy.load_state_dict(coef)
            

    model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    if (args.awp):
        proxy = nn.DataParallel(proxy, device_ids=[i for i in range(gpu_num)])
        proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
        awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)
    else:
        awp_adversary = None

    train_loader, test_loader = prepare_cifar(args.batch_size, args.test_batch_size)
    optimizer= optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    best_epoch, best_robust_acc = 0, 0.
    duration = 0
    for e in range(args.epoch):
        STARTTIME = time.time()
        adjust_learning_rate(optimizer, e, args)
        raw_adv_dist = adjust_raw_adv_coef(e, args.raw_adv_dist_coef)
        train_acc, train_robust_acc, loss = train_adv_epoch(model, args, train_loader, device,  optimizer, e,
                                                             loss_criterion=args.loss_criterion, 
                                                            with_raw_sample=args.with_raw_sample, raw_adv_coef=raw_adv_dist,
                                                            awp_adversary=awp_adversary)
        ENDTIME = time.time()
        if e%3==0 or (e>=74 and e<=80):
            test_acc, test_robust_acc, dist= eval_model_pgd( model,  test_loader, device, args.step_size, args.epsilon, args.perturb_steps)
        else:
            test_acc, _ = eval_model( model,  test_loader, device)
        if test_robust_acc > best_robust_acc:
            best_robust_acc, best_epoch = test_robust_acc, e
        if e > (args.epoch / 3):
            torch.save(model.module.state_dict(),  
             os.path.join(log_dir, f"{args.model}-e{e}-{test_acc:.4f}_{test_robust_acc:.4f}-best.pt"))
        
        duration += (ENDTIME - STARTTIME) / 3600.0
        log.print(f"Epoch:{e}, loss:{loss:.5f}, train_duration:{duration:.4f}, train_acc:{train_acc:.4f}, train_robust_acc:{train_robust_acc:.4f},  " + 
                            f"test_acc:{test_acc:.4f}, test_robust_acc:{test_robust_acc:.4f}, " +
                            f"best_robust_acc:{best_robust_acc:.4f} in epoch {best_epoch}." )
    torch.save(model.module.state_dict(), f"{log_dir}/{args.model}_e{args.epoch - 1}_{test_acc:.4f}_{test_robust_acc:.4f}-final.pt")
        
