import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def Binary_dice_loss(predictive, target, ep=1e-8, mask=None):
    if mask is not None:
        predictive = torch.masked_select(predictive, mask)
        target = torch.masked_select(target, mask)
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, mask=None, ep=1e-8, reduction='mean'):
    kl_loss=nn.KLDivLoss(reduction=reduction)
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    if mask is not None:
        # import ipdb; ipdb.set_trace()
        # consist_loss = torch.mean(consist_loss*mask.unsqueeze(dim=1))
        
        consist_loss = torch.mean(torch.masked_select(consist_loss, mask.unsqueeze(dim=1).bool()))
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2, mask=None):
    if mask is None:
        return torch.mean((input1 - input2)**2)
    else:
        mse = (input1 - input2)**2
        return torch.mean(torch.masked_select(mse, mask.bool().unsqueeze(1)))

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, mask=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            # import ipdb; ipdb.set_trace()
            if mask is not None:
                dice = self._dice_loss(torch.masked_select(inputs[:, i], mask), torch.masked_select(target[:, i], mask))
            else:
                dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
    
class GumbelTopK(nn.Module):
    """
    Perform top-k or Gumble top-k on given data
    """
    def __init__(self, k: int, dim: int = -1, gumble: bool = False):
        """
        Args:
            k: int, number of chosen
            dim: the dimension to perform top-k
            gumble: bool, whether to introduce Gumble noise when sampling
        """
        super().__init__()
        self.k = k
        self.dim = dim
        self.gumble = gumble

    def forward(self, logits):
        # logits shape: [B, N], B denotes batch size, and N denotes the multiplication of channel and spatial dim
        if self.gumble:
            u = torch.rand(size=logits.shape, device=logits.device)
            z = - torch.log(- torch.log(u))
            return torch.topk(logits + z, self.k, dim=self.dim)
        else:
            return torch.topk(logits, self.k, dim=self.dim)


class Contrast_Loss(nn.Module):
    
    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
        self.topk = GumbelTopK(k=sample_num, dim=0)
        self.bidir = bidirectional
        
    def forward(self, pred, pseudo_label, proj, avg_proj, mask, fn_mask, sample_num=5):
        '''
            pred: 24, 4, 256, 256
            pseudo_label: 24, 4, 256, 256
            proj: 24, 64, 256, 256
            avg_proj: 24, 64, 256, 256
            mask: 24, 256, 256
            fn_mask: 24, 256, 256
        '''
        loss = 0
        mask = mask.bool()
        fn_mask = fn_mask.bool()
        pred = pred.argmax(dim=1)
        pseudo_label = pseudo_label.argmax(dim=1)

        
        # B, proj_d, _, _ = proj.shape
        # for b_idx in range(B):
        #     s_mask = mask[b_idx]
        #     s_fn_mask = fn_mask[b_idx]
        #     sample_num = min(sample_num, s_mask.sum().item(), s_fn_mask.sum().item())
            
        #     s_mask_list = torch.where(s_mask==True)
        #     s_mask_idx = torch.randperm(len(s_mask_list[0]))[:sample_num]
        #     # s_mask*=False
        #     s_mask = torch.zeros_like(s_mask).to(s_mask.device)
        #     s_mask[s_mask_list[0][s_mask_idx], s_mask_list[1][s_mask_idx]]=True

        #     s_fn_mask_list = torch.where(s_fn_mask==True)
        #     s_fn_mask_idx = torch.randperm(len(s_fn_mask_list[0]))[:sample_num]
        #     # s_fn_mask*=False
        #     s_fn_mask = torch.zeros_like(s_fn_mask).to(s_fn_mask.device)
        #     s_fn_mask[s_fn_mask_list[0][s_fn_mask_idx], s_fn_mask_list[1][s_fn_mask_idx]]=True

        #     fea_source = torch.masked_select(proj[b_idx][:], s_fn_mask.unsqueeze(dim=0)).view(proj_d, -1)
        #     fea_pos = torch.masked_select(avg_proj[b_idx][:], s_fn_mask.unsqueeze(dim=0)).view(proj_d, -1)
        #     fea_neg = torch.masked_select(proj[b_idx][:], s_mask.unsqueeze(dim=0)).view(proj_d, -1)

        #     # print('fea source:', fea_source.shape, 'fea_pos:', fea_pos.shape, 'fea_neg:', fea_neg.shape)

        #     # import ipdb; ipdb.set_trace()
        #     positive = F.cosine_similarity(fea_source, fea_pos, dim=-1)  # B * N
        #     negative = F.cosine_similarity(fea_source, fea_neg, dim=-1)  # B * N

        #     nominator = torch.exp(positive/self.tau)
        #     denominator = torch.exp(negative/self.tau) + nominator
        #     loss = -torch.log(nominator / (denominator + 1e-8)).mean()
                
        # # import ipdb; ipdb.set_trace()
        


        proj_b, proj_d, _, _ = proj.shape
        for proj_b_idx in range(proj_b):
            s_mask = mask[proj_b_idx]
            s_fn_mask = fn_mask[proj_b_idx]
            # s_mask_num = s_mask.sum()
            # s_fn_mask_num = s_fn_mask.sum()
            # s_pred = pred[proj_b_idx]
            s_plabel = pseudo_label[proj_b_idx]
            s_class = sorted(s_plabel.unique().cpu().detach().numpy())

            for c_idx in s_class:
                sc_mask = s_mask&(s_plabel!=c_idx)
                # if sc_mask.sum().item() < sample_num:
                #     loss += 0
                #     continue
                sc_fn_mask = s_fn_mask&(s_plabel==c_idx)
                # if sc_fn_mask.sum().item() < sample_num:
                #     loss += 0
                #     continue

                sample_num = min(sample_num, sc_mask.sum().item(), sc_fn_mask.sum().item())

                sc_mask_list = torch.where(sc_mask==True)
                sc_mask_idx = torch.randperm(len(torch.where(sc_mask==True)[0]))[:sample_num]
                sc_mask*=False
                sc_mask[sc_mask_list[0][sc_mask_idx], sc_mask_list[1][sc_mask_idx]]=True

                # if sc_mask.sum() < 5:
                #     import ipdb; ipdb.set_trace()
                
                sc_fn_mask_list = torch.where(sc_fn_mask==True)
                # sc_fn_mask_idx = torch.randint(len(torch.where(sc_fn_mask==True)[0]), (sample_num,)) 
                sc_fn_mask_idx = torch.randperm(len(torch.where(sc_fn_mask==True)[0]))[:sample_num]
                sc_fn_mask*=False
                sc_fn_mask[sc_fn_mask_list[0][sc_fn_mask_idx], sc_fn_mask_list[1][sc_fn_mask_idx]]=True
                
                fea_source = torch.masked_select(proj[proj_b_idx][:], sc_fn_mask.unsqueeze(dim=0)).view(proj_d, -1)
                fea_pos = torch.masked_select(avg_proj[proj_b_idx][:], sc_fn_mask.unsqueeze(dim=0)).view(proj_d, -1)
                fea_neg = torch.masked_select(proj[proj_b_idx][:], sc_mask.unsqueeze(dim=0)).view(proj_d, -1)

                # import ipdb; ipdb.set_trace()
                positive = F.cosine_similarity(fea_source, fea_pos, dim=-1)  # B * N
                negative = F.cosine_similarity(fea_source, fea_neg, dim=-1)  # B * N

                nominator = torch.exp(positive/self.tau)
                denominator = torch.exp(negative/self.tau) + nominator
                loss = -torch.log(nominator / (denominator + 1e-8)).mean()
                
                # print(loss)

                # import ipdb; ipdb.set_trace()

                # print(sc_mask.sum())
                # import ipdb; ipdb.set_trace()
            

            # select_idx = torch.randperm(s_mask_num)[:sample_num]
            # fn_select_idx = torch.randint(s_fn_mask_num, (sample_num,))
            # proj_selected = torch.masked_select(proj, s_fn_mask).view(proj_b, proj_d, -1)[:, :, fn_select_idx]
            # avg_proj_selected = torch.masked_select(proj, s_fn_mask).view(proj_b, proj_d, -1)[:, :, fn_select_idx]

            # print(proj_selected.shape)
            # print(avg_proj_selected.shape)

        #     import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()
        # pass
        return loss