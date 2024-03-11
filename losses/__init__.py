

import torch

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def tv_loss(x):
    tv_loss =TVLoss().to(device)
    tv_loss = tv_loss(x)

    return tv_loss

def tv_vi (fused_result,input_vi ):
    tv_vi=torch.norm((tv_loss(fused_result)-tv_loss(input_vi)),1)

    return tv_vi


def tv_ir (fused_result,input_ir):
    tv_r=torch.norm((tv_loss(fused_result)-tv_loss(input_ir)),1)

    return tv_r

def CharbonnierLoss_IR(f,ir):
    eps = 1e-3
    loss=torch.mean(torch.sqrt((f-ir)**2+eps**2))
    return loss

def CharbonnierLoss_VI(f,vi):
    eps = 1e-3
    loss=torch.mean(torch.sqrt((f-vi)**2+eps**2))
    return loss

