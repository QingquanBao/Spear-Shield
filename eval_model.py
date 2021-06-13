import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from utils import prepare_cifar
from pgd_attack import pgd_attack
from tqdm import tqdm, trange

def topk_acc(logits, y, k:int):
    '''
        Input: 
            logits  (shape: B * N)
            y       (shape: B * 1)
            k: topK
    '''
    rightnum = torch.sum(torch.argsort(logits, dim=1)[:,-1*k:] == y.reshape(-1,1) )
    return rightnum 


def eval_model(model, test_loader, device):
    correct_adv, correct = [], []
    distance = []
    num = 0
    with trange(10000) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            model.eval()
            with torch.no_grad():
                output = model(x)
            pred = output.argmax(dim=1)
            correct.append(pred == label)
            num += x.shape[0]
            pbar.set_description(f"Acc: {torch.cat(correct).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    return natural_acc, distance

def eval_model_pgd(model,  test_loader, device, step_size, epsilon, perturb_steps, k=1):
    correct_adv, correct = [], []
    correct_k_natural, correct_k_adv = 0, 0
    distance = []
    num = 0
    with trange(10000) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            x_adv = pgd_attack(model, x.clone(), label.clone(), step_size, epsilon, perturb_steps)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
            distance.append(torch.max((x-x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            if ( k > 1):
                correct_k_natural += topk_acc(output, label, k=k)
                correct_k_adv     += topk_acc(output_adv, label, k=k)
            num += x.shape[0]
            pbar.set_description(f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f},Top_k_natural: {correct_k_natural / num :.5f}, Top_k_robust: {float(correct_k_adv / num) :.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    robust_acc = torch.cat(correct_adv).float().mean()
    distance = torch.cat(distance).max()
    if ( k > 1) :
        natural_topk_acc = correct_k_natural / num
        robust_topk_acc  = correct_k_adv     / num
    else:
        natural_topk_acc = None
        robust_topk_acc  = None
    return natural_acc, robust_acc, distance, natural_topk_acc, robust_topk_acc

def eval_model_with_attack(model, test_loader, attack, epsilon, device, k=1):
    correct_adv, correct = [], []
    correct_k_natural, correct_k_adv = 0.0 , 0.0
    distance = []
    num = 0
    with trange(10000) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            x_adv = attack(model, x.clone(), label.clone())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = x_adv.clamp(0,1)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
            distance.append(torch.max((x-x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            if ( k > 1):
                correct_k_natural += topk_acc(output, label, k=k)
                correct_k_adv     += topk_acc(output_adv, label, k=k)
            num += x.shape[0]
            rkacc = float(correct_k_adv) / float(num)
            pbar.set_description(f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f}, Top_k_natural: {correct_k_natural / num :.5f}, Top_k_robust: {float(rkacc) :.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    robust_acc = torch.cat(correct_adv).float().mean()
    distance = torch.cat(distance).max()
    if ( k > 1) :
        natural_topk_acc = correct_k_natural / num
        robust_topk_acc  = correct_k_adv     / num
    else:
        natural_topk_acc = None
        robust_topk_acc  = None
    return natural_acc, robust_acc, distance, natural_topk_acc, robust_topk_acc
