import torch
import torch.nn as nn
import torch.nn.functional as F

def margin_loss(logits,y):
    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

def pgd_attack(model, x, y, step_size, epsilon, perturb_steps,
                random_start=None, distance='l_inf'):
    model.eval()
    batch_size = len(x)
    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

class PGDAttack():
    def __init__(self, step_size, epsilon, perturb_steps, criterion='CrossEntropy',
                random_start=None, ODI_num_steps=5, ODI_step_size=0.01):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start
        self.criterion = criterion.split('+')[0]
        if (len(criterion.split('+')) == 1):
            self.mix = False
        else:
            self.mix = True
        self.odi_num_steps = ODI_num_steps
        self.odi_step_size = ODI_step_size

    def __call__(self, model, x, y, isODI=False):
        model.eval()
        if self.criterion == 'CE':
            criterion = F.cross_entropy
        elif self.criterion == 'MSE':
            criterion = F.mse_loss
        elif self.criterion == 'Margin':
            criterion = margin_loss
      
        if self.random_start:
            x_adv = x.detach() + self.random_start * torch.randn(x.shape).cuda().detach()
        else:
            x_adv = x.detach()

        out = model(x)
        #randVector_ = torch.FloatTensor(*out.shape).uniform_(-1.,1.).to(x.device)
        out_norm = F.softmax(out, dim=0)
        entroVector_ = (torch.log(out_norm) + 1 ).to(x.device) * -1  
        entroVector_ = entroVector_ / torch.linalg.norm(entroVector_)
        if isODI:
            for i in range(self.odi_num_steps):
                x_adv.requires_grad_()
                loss = (model(x_adv) * entroVector_.detach()).sum()
                grad = torch.autograd.grad(loss, [x_adv])[0]
                x_adv = x_adv.detach() + self.odi_step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Vanilla PGD
        for i in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                y_adv = model(x_adv)
                '''
                if self.random_start:
                    y_adv += torch.normal(0, self.random_start, y_adv.shape).to(y_adv.device)
                '''
                if self.mix == True:
                    y_adv_norm = F.softmax(y_adv, dim=0)
                    y_adv_entro = -1 * torch.mul(y_adv_norm, torch.log(y_adv_norm)).sum()
                    coef = 0.1 / (i+1) #0.7 * (1 - i / self.perturb_steps) #1 / (i+1) #
                    loss_c = coef * y_adv_entro  + criterion(y_adv, y)   
                else:
                    loss_c = criterion(y_adv, y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv       
   
class TransferPGD():
    def __init__(self, step_size, epsilon, perturb_steps, whitemodel,
                 criterion='CrossEntropy',
                random_start=None ):
        self.whitemodel = whitemodel
        self.attack = PGDAttack(step_size, epsilon, perturb_steps, criterion, random_start)
     
    def __call__(self, model, x, y):
        x_adv = self.attack(self.whitemodel, x, y)
        return x_adv

class NaiveAttack():
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, model, x, y):
        perturb = torch.arange(0, 256*3*32*32).reshape(x.shape)
        perturb = (perturb % 2 ==0) * 2 * self.epsilon  - self.epsilon
        perturb = perturb.to(x.device)
        x_adv = x.detach() + perturb
        return x_adv
