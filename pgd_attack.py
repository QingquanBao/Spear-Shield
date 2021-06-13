import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def margin_loss(logits,y):
    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss

def pgd_attack(model, x, y, step_size, epsilon, perturb_steps, criterion='CE',
                random_start=None, distance='l_inf'):
    model.eval()
    batch_size = len(x)
    if criterion == 'CE':
        criterion = F.cross_entropy
    elif criterion == 'MSE':
        criterion = F.mse_loss
    elif criterion == 'Margin':
        criterion = margin_loss

    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = criterion(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def weak_pgd_attack(model, x, y, step_size, epsilon, perturb_steps, criterion='CE',
                random_start=None, distance='l_inf'):
    model.eval()
    batch_size = len(x)
    if criterion == 'CE':
        criterion = F.cross_entropy
    elif criterion == 'MSE':
        criterion = F.mse_loss
    elif criterion == 'Margin':
        criterion = margin_loss

    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = criterion(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach()) + random_start * torch.randn(x.shape).cuda().detach()

            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def Trades_perturb(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf'):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1),
                                   reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise ValueError('Only support l-inf distance now, u give {}'.format(distance))

    return x_adv

class ODIAttack():
    def __init__(self, 
                  epsilon,
                  num_steps,
                  step_size,
                  ODI_num_steps=10,
                  ODI_step_size=0.02,
                  random=None
                ):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.ODI_num_steps=ODI_num_steps
        self.ODI_step_size=ODI_step_size
        self.random = random
        
    def __call__(self,model,
                  X,
                  y
                  ):
        model.eval()
        out = model(X)
        device = X.device
        acc_clean = (out.data.max(1)[1] == y.data).float().sum()
        X_final = Variable(X.data, requires_grad=False, device=device).clone()

        restart_num=4
        for idx in range(restart_num):
            X_pgd = Variable(X.data, requires_grad=True, device=device).clone()
            randVector_ = torch.FloatTensor(*out.shape).uniform_(-1.,1.).to(device)
            '''
            randVector_ = torch.rand(*out.shape, device=device, requires_grad=True)
            sampling_num = 10
            for _ in range(sampling_num):
                with torch.enable_grad():
                    randVector_.requires_grad_()
                    smalloss = nn.CrossEntropyLoss()(randVector_, y)
                    grad = torch.autograd.grad(smalloss, [randVector_])[0]
                    randVector_ = randVector_.detach() - 0.01 * grad.detach() + torch.sqrt(torch.tensor(2*0.01)).to(device).detach() * torch.randn(*randVector_.shape).to(device).detach()
            randVector_ = randVector_.detach()
            out_norm = F.softmax(out, dim=0)
            randVector_ = (torch.log(out_norm) + 1 ).to(device) * -1  
            randVector_ = (randVector_ / torch.linalg.norm(randVector_)).detach()
            '''
            if self.random:
                random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-self.epsilon, self.epsilon).to(device)
                X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

            for i in range(self.ODI_num_steps + self.num_steps):
                opt = optim.SGD([X_pgd], lr=1e-3)
                opt.zero_grad()
                with torch.enable_grad():
                    if i < self.ODI_num_steps:
                        loss = (model(X_pgd) * randVector_).sum()
                    else:
                        loss = margin_loss(model(X_pgd),y)
                        #loss = nn.CrossEntropyLoss()(model(X_pgd),y)
                loss.backward()
                if i < self.ODI_num_steps: 
                    eta = self.ODI_step_size * X_pgd.grad.data.sign()
                else:
                    eta = self.step_size * X_pgd.grad.data.sign()
                X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                eta = torch.clamp(X_pgd.data - X.data, -self.epsilon, self.epsilon)
                X_pgd = Variable(X.data + eta, requires_grad=True)
                X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
            
            indices = (torch.argmax(model(X_pgd), dim=1) != y)
            X_final[indices,:,:,:] = X_pgd[indices,:,:,:]

        return X_final


class PGDAttack():
    def __init__(self, step_size, epsilon, perturb_steps, criterion='CE',
                random_start=None, ODI_num_steps=20, ODI_step_size=0.007,
                ODI=False, mask=False):
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
        self.ODI = ODI
        self.mask = mask
        

    def __call__(self, model, x, y):
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
        '''
        out_norm = F.softmax(out, dim=0)
        entroVector_ = (torch.log(out_norm) + 1 ).to(x.device) * -1  
        entroVector_ = entroVector_ / torch.linalg.norm(entroVector_)
        '''
        if self.ODI:
            #entroVector_ = torch.FloatTensor(*out.shape).uniform_(-1.,1.).to(x.device)
            '''
            _, top2indices = torch.topk(out, k=2, dim=1)
            entroVector_ = out.clone()
            entroVector_[:,top2indices[:,0]], entroVector_[:,top2indices[:,1]] = out[:,top2indices[:,1]], out[:,top2indices[:,0]]
            '''
            entroVector_ = out.sum(dim=1, keepdim=True).detach()
            for i in range(self.odi_num_steps):
                x_adv.requires_grad_()
                loss = (model(x_adv) * entroVector_.detach()).sum()
                grad = torch.autograd.grad(loss, [x_adv])[0]
                x_adv = x_adv.detach() + self.odi_step_size * (grad.detach())
                x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        mask = torch.cat((
                        torch.zeros((1,x.shape[1],x.shape[2],x.shape[3])),
                        torch.ones((1,x.shape[1],x.shape[2],x.shape[3]))
                        )).cuda()
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
                    loss_c = criterion(y_adv, y) + coef * y_adv_entro     
                else:
                    loss_c =  criterion(y_adv, y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            if (self.mask == True and i >= 0):
                pred = y_adv.argmax(dim=1)
                tmp = (pred == y).to(torch.int64)
                grad_mask = mask[tmp]
                x_adv = x_adv.detach() + grad_mask.detach() * self.step_size * grad.detach()  \
                                       + grad_mask.detach() * 0.1 * torch.sqrt(2 * torch.tensor(self.step_size)).cuda() * torch.randn(x.shape).cuda().detach()
            else:
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())  
                 #+ self.random_start * torch.randn(x.shape).cuda().detach()
                             #+  torch.sqrt(2 * torch.tensor(self.step_size)).cuda() * torch.randn(x.shape).cuda().detach()

            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv       
   
class TradesAttack():
    def __init__(self,
              step_size=0.003,
              epsilon=0.031,
              perturb_steps=10,
              random_start = 0.001,
              distance='l_inf'):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.random_start = random_start 

    def __call__(self, model, x_natural, y): 
        model.eval()
        device = x_natural.device
        batch_size = len(x_natural)
        if self.distance == 'l_inf':
            x_adv = x_natural.detach() + self.random_start * torch.randn(x_natural.shape).to(device).detach()
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError('Only support l-inf distance now, u give {}'.format(distance))

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
