import torch
import torch.nn as nn
import torch.nn.functional as F

# loss_contrast += contrast_loss_fn(y_ori, proj_list, i, y_pseudo_label[0], mask=y_mask[0], sample_num=400)
class Contrast1(nn.Module):

    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super(Contrast1, self).__init__()
        self.tau = temperature
        # self.sample_num = sample_num
        # self.topk = GumbelTopK(k=sample_num, dim=0)
        # self.bidir = bidirectional

    # def forward(self, pred, proj_list, idx, pseudo_label, mask, sample_num=5):
    def forward(self, proj_list, idx, pseudo_label, mask, sample_num=5):
        '''
            mask为确定区域

            batch_size: 24
            pred: 3, 24, 4, 256, 256
            proj_list: (24, 64, 256, 256), (24, 64, 256, 256), (24, 64, 256, 256)
            idx = 0
            pseudo_label: 24, 256, 256
            mask: 24, 256, 256
        '''
        batch_size = mask.shape[0]
        loss = 0
        
        curr_proj = None
        pos_proj = []
        for i in range(len(proj_list)):
            proj = proj_list[i].permute(0, 2, 3, 1)
            # 24, 65536, 64
            proj = proj.contiguous().view(proj.shape[0], -1, proj.shape[-1])
            if i == idx:
                # curr_proj = proj
                curr_proj = F.normalize(proj, dim=-1)
            else:
                # pos_proj.append(proj.unsqueeze(1))
                pos_proj.append(F.normalize(proj.unsqueeze(1), dim=-1))
        pos_proj = torch.cat(pos_proj, dim=1)

        # pseudo_label = pseudo_label.contiguous().view(batch_size, -1)
        mask = mask.contiguous().view(batch_size, -1)
        fn_mask = 1-mask
        
        for b_idx in range(batch_size):
            mask_ = mask[b_idx]
            fn_mask_ = fn_mask[b_idx]
            c_proj = curr_proj[b_idx]
            p_proj = pos_proj[b_idx]

            easy_indices = mask_.nonzero()
            hard_indices = fn_mask_.nonzero()

            num_easy = easy_indices.shape[0]
            num_hard = hard_indices.shape[0]

            hard_sample_num = min(sample_num//2, num_hard)
            easy_sample_num = min(sample_num-hard_sample_num, num_easy)

            easy_perm = torch.randperm(num_easy)
            easy_indices = easy_indices[easy_perm[:easy_sample_num]]
            hard_perm = torch.randperm(num_hard)
            hard_indices = hard_indices[hard_perm[:hard_sample_num]]
            indices = torch.cat((hard_indices, easy_indices), dim=0)

            c_proj_selected = c_proj[indices].squeeze(dim=1)
            p_proj_selected = p_proj[:, indices].squeeze(dim=2)


            # pos_loss_item = 0
            # for p_proj_idx in range(p_proj_selected.shape[0]):
            #     for sample_idx in range(len(c_proj_selected)):
            #         pos_loss_item += torch.exp(torch.div(torch.matmul(c_proj_selected[sample_idx], p_proj_selected[0][sample_idx]), self.tau))

            # mul_martix = torch.matmul(c_proj_selected, c_proj_selected.T)
            #     # import ipdb; ipdb.set_trace()
            # neg_loss_item = torch.exp(torch.div(torch.masked_select(mul_martix, (1-torch.eye(mul_martix.shape[0])).bool()), self.tau)).sum()

            # pos_loss_item = 0
            # neg_loss_item = 0
            # for sample_idx in range(c_proj_selected.shape[0]):
            #     # 0-399 sample_num
            #     for outputs_idx in range(len(p_proj_selected)):
            #         # pos_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], p_proj_selected[outputs_idx][sample_idx], dim=-1) / self.tau)
            #         pos_loss_item += F.cosine_similarity(c_proj_selected[sample_idx], p_proj_selected[outputs_idx][sample_idx], dim=-1)
            # print('item:', pos_loss_item)
            # print('sum:', F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum())

            pos_loss_item = F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum(0)
            pos_loss_item = torch.exp(pos_loss_item / self.tau)
            # mask = (1 - torch.eye(matrix.shape[0], device=matrix.device)).bool()
            # neg_loss_item = torch.masked_select(matrix, mask)
            # matrix.sum(dim=0)-torch.diagonal(matrix)
            matrix = F.cosine_similarity(c_proj_selected.unsqueeze(dim=1), c_proj_selected.unsqueeze(dim=0), dim=-1)
            matrix = torch.exp(matrix / self.tau)
            neg_loss_item = matrix.sum(dim=0) - torch.diagonal(matrix)

            loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()

                
                # for sample_idx_ in range(c_proj_selected.shape[0]):
                #     if sample_idx !=  sample_idx_:
                #         neg_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], c_proj_selected[sample_idx_], dim=-1) / self.tau)
                #         # import ipdb; ipdb.set_trace()
                # print(pos_loss_item, neg_loss_item)

            # nominator = pos_loss_item
            # denominator = neg_loss_item + nominator
            # loss += -torch.log(nominator / (denominator + 1e-8)).mean()
        return loss / batch_size
            # import ipdb; ipdb.set_trace()

        # easy_indices = mask.nonzero()
        # hard_indices = fn_mask.nonzero()

        # num_easy = easy_indices.shape[0]
        # num_hard = hard_indices.shape[0]

        # hard_sample_num = min(sample_num//2, num_hard)
        # easy_sample_num = min(sample_num-hard_sample_num, num_easy)

        # easy_perm = torch.randperm(num_easy)
        # easy_indices = easy_indices[easy_perm[:easy_sample_num]]
        # hard_perm = torch.randperm(num_hard)
        # hard_indices = hard_indices[hard_perm[:hard_sample_num]]
        # indices = torch.cat((hard_indices, easy_indices), dim=0)


        # import ipdb; ipdb.set_trace()

        # labels = labels.contiguous().view(batch_size, -1)
        # predict = predict.contiguous().view(batch_size, -1)
        # feats = feats.permute(0, 2, 3, 1)
        # feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        # feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        # loss = self._contrastive(feats_, labels_)

        # pass


class Contrast2(nn.Module):

    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super(Contrast2, self).__init__()
        self.tau = temperature

    def forward(self, proj_list, idx, pseudo_label, mask, sample_num=5):
        '''
            mask为确定区域

            batch_size: 24
            pred: 3, 24, 4, 256, 256
            proj_list: (24, 64, 256, 256), (24, 64, 256, 256), (24, 64, 256, 256)
            idx = 0
            pseudo_label: 24, 256, 256
            mask: 24, 256, 256
        '''
        batch_size = mask.shape[0]
        loss = 0
        
        curr_proj = None
        pos_proj = []
        for i in range(len(proj_list)):
            try:
                proj = proj_list[i].permute(0, 2, 3, 1)
            except:
                proj = proj_list[i].permute(0, 2, 3, 4, 1)

            proj = proj.contiguous().view(proj.shape[0], -1, proj.shape[-1])
            if i == idx:
                curr_proj = proj
                # curr_proj = F.normalize(proj, dim=-1)
            else:
                pos_proj.append(proj.unsqueeze(1))
                # pos_proj.append(F.normalize(proj.unsqueeze(1), dim=-1))
        pos_proj = torch.cat(pos_proj, dim=1)
        # import ipdb; ipdb.set_trace()
    

        mask = mask.contiguous().view(batch_size, -1).long()
        fn_mask = 1-mask
        
        for b_idx in range(batch_size):
            # mask_ = mask[b_idx]
            fn_mask_ = fn_mask[b_idx]
            c_proj = curr_proj[b_idx]
            p_proj = pos_proj[b_idx]

            hard_indices = fn_mask_.nonzero()

            num_hard = hard_indices.shape[0]

            hard_sample_num = min(sample_num, num_hard)

            hard_perm = torch.randperm(num_hard)
            hard_indices = hard_indices[hard_perm[:hard_sample_num]]
            indices = hard_indices

            # import ipdb; ipdb.set_trace()
            c_proj_selected = c_proj[indices].squeeze(dim=1)
            p_proj_selected = p_proj[:, indices].squeeze(dim=2)

            c_proj_selected = F.normalize(c_proj_selected, dim=-1)
            p_proj_selected = F.normalize(p_proj_selected, dim=-1)

            pos_loss_item = F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum(0)
            pos_loss_item = torch.exp(pos_loss_item / self.tau)
            matrix = F.cosine_similarity(c_proj_selected.unsqueeze(dim=1), c_proj_selected.unsqueeze(dim=0), dim=-1)
            matrix = torch.exp(matrix / self.tau)
            neg_loss_item = matrix.sum(dim=0) - torch.diagonal(matrix)

            loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()

        return loss / batch_size


class Contrast3(nn.Module):

    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super(Contrast3, self).__init__()
        self.tau = temperature
        # self.sample_num = sample_num
        sample_num = 400
        # self.topk = GumbelTopK(k=sample_num, dim=0)
        # self.bidir = bidirectional

    def forward(self, pred, proj_list, idx, pseudo_label, mask, sample_num=5):
    # def forward(self, proj_list, idx, pseudo_label, mask, sample_num=5):
        '''
            mask为确定区域

            batch_size: 24
            pred: 3, 24, 4, 256, 256
            proj_list: (24, 64, 256, 256), (24, 64, 256, 256), (24, 64, 256, 256)
            idx = 0
            pseudo_label: 24, 256, 256
            mask: 24, 256, 256
        '''
        batch_size = mask.shape[0]
        loss = 0

        uncertainty = torch.sum(pred*torch.log(pred+1e-6), dim=1, keepdim=True)
        
        curr_proj = None
        pos_proj = []
        for i in range(len(proj_list)):
            proj = proj_list[i].permute(0, 2, 3, 1)
            # 24, 65536, 64
            proj = proj.contiguous().view(proj.shape[0], -1, proj.shape[-1])
            if i == idx:
                # curr_proj = proj
                curr_proj = F.normalize(proj, dim=-1)
            else:
                # pos_proj.append(proj.unsqueeze(1))
                pos_proj.append(F.normalize(proj.unsqueeze(1), dim=-1))
        pos_proj = torch.cat(pos_proj, dim=1)

        # pseudo_label = pseudo_label.contiguous().view(batch_size, -1)
        mask = mask.contiguous().view(batch_size, -1)
        uncertainty = uncertainty.contiguous().view(batch_size, -1)
        fn_mask = 1-mask
        
        for b_idx in range(batch_size):
            mask_ = mask[b_idx]
            fn_mask_ = fn_mask[b_idx]
            c_proj = curr_proj[b_idx]
            p_proj = pos_proj[b_idx]
            # c_pred = pred[b_idx]
            uncertainty_ = uncertainty[b_idx]
            

            easy_indices = mask_.nonzero()
            hard_indices = fn_mask_.nonzero()

            num_easy = easy_indices.shape[0]
            num_hard = hard_indices.shape[0]

            hard_sample_num = min(sample_num//2, num_hard)
            easy_sample_num = min(sample_num-hard_sample_num, num_easy)


            # import ipdb; ipdb.set_trace()
            hard_topk = torch.topk(uncertainty_[hard_indices[:, 0]], hard_sample_num).indices
            hard_indices = hard_indices[hard_topk]
            easy_topk = torch.topk(uncertainty_[easy_indices[:, 0]], easy_sample_num).indices
            easy_indices = easy_indices[easy_topk]



            # easy_perm = torch.randperm(num_easy)
            # easy_indices = easy_indices[easy_perm[:easy_sample_num]]
            # hard_perm = torch.randperm(num_hard)
            # hard_indices = hard_indices[hard_perm[:hard_sample_num]]
            indices = torch.cat((hard_indices, easy_indices), dim=0)

            c_proj_selected = c_proj[indices].squeeze(dim=1)
            p_proj_selected = p_proj[:, indices].squeeze(dim=2)


            # pos_loss_item = 0
            # for p_proj_idx in range(p_proj_selected.shape[0]):
            #     for sample_idx in range(len(c_proj_selected)):
            #         pos_loss_item += torch.exp(torch.div(torch.matmul(c_proj_selected[sample_idx], p_proj_selected[0][sample_idx]), self.tau))

            # mul_martix = torch.matmul(c_proj_selected, c_proj_selected.T)
            #     # import ipdb; ipdb.set_trace()
            # neg_loss_item = torch.exp(torch.div(torch.masked_select(mul_martix, (1-torch.eye(mul_martix.shape[0])).bool()), self.tau)).sum()

            # pos_loss_item = 0
            # neg_loss_item = 0
            # for sample_idx in range(c_proj_selected.shape[0]):
            #     # 0-399 sample_num
            #     for outputs_idx in range(len(p_proj_selected)):
            #         # pos_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], p_proj_selected[outputs_idx][sample_idx], dim=-1) / self.tau)
            #         pos_loss_item += F.cosine_similarity(c_proj_selected[sample_idx], p_proj_selected[outputs_idx][sample_idx], dim=-1)
            # print('item:', pos_loss_item)
            # print('sum:', F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum())

            pos_loss_item = F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum(0)
            pos_loss_item = torch.exp(pos_loss_item / self.tau)
            # mask = (1 - torch.eye(matrix.shape[0], device=matrix.device)).bool()
            # neg_loss_item = torch.masked_select(matrix, mask)
            # matrix.sum(dim=0)-torch.diagonal(matrix)
            matrix = F.cosine_similarity(c_proj_selected.unsqueeze(dim=1), c_proj_selected.unsqueeze(dim=0), dim=-1)
            matrix = torch.exp(matrix / self.tau)
            neg_loss_item = matrix.sum(dim=0) - torch.diagonal(matrix)

            loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()

                
                # for sample_idx_ in range(c_proj_selected.shape[0]):
                #     if sample_idx !=  sample_idx_:
                #         neg_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], c_proj_selected[sample_idx_], dim=-1) / self.tau)
                #         # import ipdb; ipdb.set_trace()
                # print(pos_loss_item, neg_loss_item)

            # nominator = pos_loss_item
            # denominator = neg_loss_item + nominator
            # loss += -torch.log(nominator / (denominator + 1e-8)).mean()
        return loss / batch_size
            # import ipdb; ipdb.set_trace()

        # easy_indices = mask.nonzero()
        # hard_indices = fn_mask.nonzero()

        # num_easy = easy_indices.shape[0]
        # num_hard = hard_indices.shape[0]

        # hard_sample_num = min(sample_num//2, num_hard)
        # easy_sample_num = min(sample_num-hard_sample_num, num_easy)

        # easy_perm = torch.randperm(num_easy)
        # easy_indices = easy_indices[easy_perm[:easy_sample_num]]
        # hard_perm = torch.randperm(num_hard)
        # hard_indices = hard_indices[hard_perm[:hard_sample_num]]
        # indices = torch.cat((hard_indices, easy_indices), dim=0)


        # import ipdb; ipdb.set_trace()

        # labels = labels.contiguous().view(batch_size, -1)
        # predict = predict.contiguous().view(batch_size, -1)
        # feats = feats.permute(0, 2, 3, 1)
        # feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        # feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        # loss = self._contrastive(feats_, labels_)

        # pass



class Contrast4(nn.Module):

    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super(Contrast4, self).__init__()
        self.tau = temperature
        # self.sample_num = sample_num
        sample_num = 400
        # self.topk = GumbelTopK(k=sample_num, dim=0)
        # self.bidir = bidirectional

    def forward(self, pred, proj_list, idx, pseudo_label, mask, sample_num=5):
    # def forward(self, proj_list, idx, pseudo_label, mask, sample_num=5):
        '''
            mask为确定区域

            batch_size: 24
            pred: 3, 24, 4, 256, 256
            proj_list: (24, 64, 256, 256), (24, 64, 256, 256), (24, 64, 256, 256)
            idx = 0
            pseudo_label: 24, 256, 256
            mask: 24, 256, 256
        '''
        batch_size = mask.shape[0]
        loss = 0

        uncertainty = torch.sum(pred*torch.log(pred+1e-6), dim=1, keepdim=True)
        
        curr_proj = None
        pos_proj = []
        for i in range(len(proj_list)):
            proj = proj_list[i].permute(0, 2, 3, 1)
            # 24, 65536, 64
            proj = proj.contiguous().view(proj.shape[0], -1, proj.shape[-1])
            if i == idx:
                # curr_proj = proj
                curr_proj = F.normalize(proj, dim=-1)
            else:
                # pos_proj.append(proj.unsqueeze(1))
                pos_proj.append(F.normalize(proj.unsqueeze(1), dim=-1))
        pos_proj = torch.cat(pos_proj, dim=1)

        # pseudo_label = pseudo_label.contiguous().view(batch_size, -1)
        mask = mask.contiguous().view(batch_size, -1)
        uncertainty = uncertainty.contiguous().view(batch_size, -1)
        fn_mask = 1-mask
        
        for b_idx in range(batch_size):
            mask_ = mask[b_idx]
            fn_mask_ = fn_mask[b_idx]
            c_proj = curr_proj[b_idx]
            p_proj = pos_proj[b_idx]
            # c_pred = pred[b_idx]
            uncertainty_ = uncertainty[b_idx]
            

            hard_indices = fn_mask_.nonzero()

            num_hard = hard_indices.shape[0]

            hard_sample_num = min(sample_num, num_hard)


            # import ipdb; ipdb.set_trace()
            hard_topk = torch.topk(uncertainty_[hard_indices[:, 0]], hard_sample_num).indices
            indices = hard_indices[hard_topk]



            # easy_perm = torch.randperm(num_easy)
            # easy_indices = easy_indices[easy_perm[:easy_sample_num]]
            # hard_perm = torch.randperm(num_hard)
            # hard_indices = hard_indices[hard_perm[:hard_sample_num]]
            # indices = torch.cat((hard_indices, easy_indices), dim=0)

            c_proj_selected = c_proj[indices].squeeze(dim=1)
            p_proj_selected = p_proj[:, indices].squeeze(dim=2)


            # pos_loss_item = 0
            # for p_proj_idx in range(p_proj_selected.shape[0]):
            #     for sample_idx in range(len(c_proj_selected)):
            #         pos_loss_item += torch.exp(torch.div(torch.matmul(c_proj_selected[sample_idx], p_proj_selected[0][sample_idx]), self.tau))

            # mul_martix = torch.matmul(c_proj_selected, c_proj_selected.T)
            #     # import ipdb; ipdb.set_trace()
            # neg_loss_item = torch.exp(torch.div(torch.masked_select(mul_martix, (1-torch.eye(mul_martix.shape[0])).bool()), self.tau)).sum()

            # pos_loss_item = 0
            # neg_loss_item = 0
            # for sample_idx in range(c_proj_selected.shape[0]):
            #     # 0-399 sample_num
            #     for outputs_idx in range(len(p_proj_selected)):
            #         # pos_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], p_proj_selected[outputs_idx][sample_idx], dim=-1) / self.tau)
            #         pos_loss_item += F.cosine_similarity(c_proj_selected[sample_idx], p_proj_selected[outputs_idx][sample_idx], dim=-1)
            # print('item:', pos_loss_item)
            # print('sum:', F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum())

            pos_loss_item = F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum(0)
            pos_loss_item = torch.exp(pos_loss_item / self.tau)
            # mask = (1 - torch.eye(matrix.shape[0], device=matrix.device)).bool()
            # neg_loss_item = torch.masked_select(matrix, mask)
            # matrix.sum(dim=0)-torch.diagonal(matrix)
            matrix = F.cosine_similarity(c_proj_selected.unsqueeze(dim=1), c_proj_selected.unsqueeze(dim=0), dim=-1)
            matrix = torch.exp(matrix / self.tau)
            neg_loss_item = matrix.sum(dim=0) - torch.diagonal(matrix)

            loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()

                
                # for sample_idx_ in range(c_proj_selected.shape[0]):
                #     if sample_idx !=  sample_idx_:
                #         neg_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], c_proj_selected[sample_idx_], dim=-1) / self.tau)
                #         # import ipdb; ipdb.set_trace()
                # print(pos_loss_item, neg_loss_item)

            # nominator = pos_loss_item
            # denominator = neg_loss_item + nominator
            # loss += -torch.log(nominator / (denominator + 1e-8)).mean()
        return loss / batch_size
            # import ipdb; ipdb.set_trace()

        # easy_indices = mask.nonzero()
        # hard_indices = fn_mask.nonzero()

        # num_easy = easy_indices.shape[0]
        # num_hard = hard_indices.shape[0]

        # hard_sample_num = min(sample_num//2, num_hard)
        # easy_sample_num = min(sample_num-hard_sample_num, num_easy)

        # easy_perm = torch.randperm(num_easy)
        # easy_indices = easy_indices[easy_perm[:easy_sample_num]]
        # hard_perm = torch.randperm(num_hard)
        # hard_indices = hard_indices[hard_perm[:hard_sample_num]]
        # indices = torch.cat((hard_indices, easy_indices), dim=0)


        # import ipdb; ipdb.set_trace()

        # labels = labels.contiguous().view(batch_size, -1)
        # predict = predict.contiguous().view(batch_size, -1)
        # feats = feats.permute(0, 2, 3, 1)
        # feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        # feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        # loss = self._contrastive(feats_, labels_)

        # pass


class Contrast5(nn.Module):

    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super(Contrast5, self).__init__()
        self.tau = temperature
        # self.sample_num = sample_num
        sample_num = 400
        # self.topk = GumbelTopK(k=sample_num, dim=0)
        # self.bidir = bidirectional

    def forward(self, pred, proj_list, idx, pseudo_label, mask, sample_num=5):
    # def forward(self, proj_list, idx, pseudo_label, mask, sample_num=5):
        '''
            mask为确定区域

            batch_size: 24
            pred: 3, 24, 4, 256, 256
            proj_list: (24, 64, 256, 256), (24, 64, 256, 256), (24, 64, 256, 256)
            idx = 0
            pseudo_label: 24, 256, 256
            mask: 24, 256, 256
        '''
        batch_size = mask.shape[0]
        loss = 0

        uncertainty = torch.sum(pred*torch.log(pred+1e-6), dim=1, keepdim=True)
        
        curr_proj = None
        pos_proj = []
        for i in range(len(proj_list)):
            proj = proj_list[i].permute(0, 2, 3, 1)
            # 24, 65536, 64
            proj = proj.contiguous().view(proj.shape[0], -1, proj.shape[-1])
            if i == idx:
                # curr_proj = proj
                curr_proj = F.normalize(proj, dim=-1)
            else:
                # pos_proj.append(proj.unsqueeze(1))
                pos_proj.append(F.normalize(proj.unsqueeze(1), dim=-1))
        pos_proj = torch.cat(pos_proj, dim=1)

        # pseudo_label = pseudo_label.contiguous().view(batch_size, -1)
        # mask = mask.contiguous().view(batch_size, -1)
        uncertainty = uncertainty.contiguous().view(batch_size, -1)
        # fn_mask = 1-mask
        
        for b_idx in range(batch_size):
            # mask_ = mask[b_idx]
            # fn_mask_ = fn_mask[b_idx]
            c_proj = curr_proj[b_idx]
            p_proj = pos_proj[b_idx]
            # c_pred = pred[b_idx]
            uncertainty_ = uncertainty[b_idx]
            

            # hard_indices = fn_mask_.nonzero()

            # num_hard = hard_indices.shape[0]

            # hard_sample_num = min(sample_num, num_hard)


            # import ipdb; ipdb.set_trace()
            indices = torch.topk(uncertainty_, sample_num).indices
            # indices = hard_indices[topk]



            # easy_perm = torch.randperm(num_easy)
            # easy_indices = easy_indices[easy_perm[:easy_sample_num]]
            # hard_perm = torch.randperm(num_hard)
            # hard_indices = hard_indices[hard_perm[:hard_sample_num]]
            # indices = torch.cat((hard_indices, easy_indices), dim=0)

            c_proj_selected = c_proj[indices].squeeze(dim=1)
            p_proj_selected = p_proj[:, indices].squeeze(dim=2)


            # pos_loss_item = 0
            # for p_proj_idx in range(p_proj_selected.shape[0]):
            #     for sample_idx in range(len(c_proj_selected)):
            #         pos_loss_item += torch.exp(torch.div(torch.matmul(c_proj_selected[sample_idx], p_proj_selected[0][sample_idx]), self.tau))

            # mul_martix = torch.matmul(c_proj_selected, c_proj_selected.T)
            #     # import ipdb; ipdb.set_trace()
            # neg_loss_item = torch.exp(torch.div(torch.masked_select(mul_martix, (1-torch.eye(mul_martix.shape[0])).bool()), self.tau)).sum()

            # pos_loss_item = 0
            # neg_loss_item = 0
            # for sample_idx in range(c_proj_selected.shape[0]):
            #     # 0-399 sample_num
            #     for outputs_idx in range(len(p_proj_selected)):
            #         # pos_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], p_proj_selected[outputs_idx][sample_idx], dim=-1) / self.tau)
            #         pos_loss_item += F.cosine_similarity(c_proj_selected[sample_idx], p_proj_selected[outputs_idx][sample_idx], dim=-1)
            # print('item:', pos_loss_item)
            # print('sum:', F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum())

            pos_loss_item = F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1).sum(0)
            pos_loss_item = torch.exp(pos_loss_item / self.tau)
            # mask = (1 - torch.eye(matrix.shape[0], device=matrix.device)).bool()
            # neg_loss_item = torch.masked_select(matrix, mask)
            # matrix.sum(dim=0)-torch.diagonal(matrix)
            matrix = F.cosine_similarity(c_proj_selected.unsqueeze(dim=1), c_proj_selected.unsqueeze(dim=0), dim=-1)
            matrix = torch.exp(matrix / self.tau)
            neg_loss_item = matrix.sum(dim=0) - torch.diagonal(matrix)

            loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()

                
                # for sample_idx_ in range(c_proj_selected.shape[0]):
                #     if sample_idx !=  sample_idx_:
                #         neg_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], c_proj_selected[sample_idx_], dim=-1) / self.tau)
                #         # import ipdb; ipdb.set_trace()
                # print(pos_loss_item, neg_loss_item)

            # nominator = pos_loss_item
            # denominator = neg_loss_item + nominator
            # loss += -torch.log(nominator / (denominator + 1e-8)).mean()
        return loss / batch_size
            # import ipdb; ipdb.set_trace()

        # easy_indices = mask.nonzero()
        # hard_indices = fn_mask.nonzero()

        # num_easy = easy_indices.shape[0]
        # num_hard = hard_indices.shape[0]

        # hard_sample_num = min(sample_num//2, num_hard)
        # easy_sample_num = min(sample_num-hard_sample_num, num_easy)

        # easy_perm = torch.randperm(num_easy)
        # easy_indices = easy_indices[easy_perm[:easy_sample_num]]
        # hard_perm = torch.randperm(num_hard)
        # hard_indices = hard_indices[hard_perm[:hard_sample_num]]
        # indices = torch.cat((hard_indices, easy_indices), dim=0)


        # import ipdb; ipdb.set_trace()

        # labels = labels.contiguous().view(batch_size, -1)
        # predict = predict.contiguous().view(batch_size, -1)
        # feats = feats.permute(0, 2, 3, 1)
        # feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        # feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        # loss = self._contrastive(feats_, labels_)

        # pass


class Contrast6(nn.Module):

    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super(Contrast6, self).__init__()
        self.tau = temperature
        # self.sample_num = sample_num
        # self.topk = GumbelTopK(k=sample_num, dim=0)
        # self.bidir = bidirectional

    # def forward(self, pred, proj_list, idx, pseudo_label, mask, sample_num=5):
    def forward(self, proj_list, idx, pseudo_label, mask, sample_num=5):
        '''
            mask为确定区域

            batch_size: 24
            pred: 3, 24, 4, 256, 256
            proj_list: (24, 64, 256, 256), (24, 64, 256, 256), (24, 64, 256, 256)
            idx = 0
            pseudo_label: 24, 256, 256
            mask: 24, 256, 256
        '''
        batch_size = mask.shape[0]
        loss = 0
        
        curr_proj = None
        pos_proj = []
        for i in range(len(proj_list)):
            proj = proj_list[i].permute(0, 2, 3, 1)
            # 24, 65536, 64
            proj = proj.contiguous().view(proj.shape[0], -1, proj.shape[-1])
            if i == idx:
                # curr_proj = proj
                curr_proj = F.normalize(proj, dim=-1)
            else:
                # pos_proj.append(proj.unsqueeze(1))
                pos_proj.append(F.normalize(proj.unsqueeze(1), dim=-1))
        pos_proj = torch.cat(pos_proj, dim=1)

        pseudo_label = pseudo_label.contiguous().view(batch_size, -1)
        mask = mask.contiguous().view(batch_size, -1)
        fn_mask = 1-mask
        
        for b_idx in range(batch_size):
            mask_ = mask[b_idx]
            fn_mask_ = fn_mask[b_idx]
            c_proj = curr_proj[b_idx]
            p_proj = pos_proj[b_idx]
            pseudo_label_ = pseudo_label[b_idx]

            hard_indices = fn_mask_.nonzero()

            num_hard = hard_indices.shape[0]

            hard_sample_num = min(sample_num, num_hard)

            hard_perm = torch.randperm(num_hard)
            hard_indices = hard_indices[hard_perm[:hard_sample_num]]
            # indices = torch.cat((hard_indices, easy_indices), dim=0)
            indices = hard_indices

            c_proj_selected = c_proj[indices].squeeze(dim=1)
            p_proj_selected = p_proj[:, indices].squeeze(dim=2)

            pseudo_label_selected = pseudo_label_[indices].squeeze(dim=1)
            
            # mask = (1 - torch.eye(matrix.shape[0], device=matrix.device)).bool()
            # neg_loss_item = torch.masked_select(matrix, mask)
            # matrix.sum(dim=0)-torch.diagonal(matrix)

            # neg_loss_item = 0
            p_label_unique = pseudo_label_selected.unique()
            # for p_label in pseudo_label_selected.unique().cpu().detach().numpy():
            for p_label_unique_idx in range(len(p_label_unique)):
                curr_label = p_label_unique[p_label_unique_idx]

                # pos
                pos_loss_item = F.cosine_similarity(c_proj_selected[pseudo_label_selected==curr_label], p_proj_selected[:, pseudo_label_selected==curr_label], dim=-1).sum(0)
                pos_loss_item = torch.exp(pos_loss_item / self.tau)

                # neg
                matrix = F.cosine_similarity(c_proj_selected[pseudo_label_selected==curr_label].unsqueeze(dim=1), c_proj_selected[pseudo_label_selected!=curr_label].unsqueeze(dim=0), dim=-1)
                matrix = torch.exp(matrix / self.tau)
                neg_loss_item = matrix.sum(-1)
                # import ipdb; ipdb.set_trace()
                loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()

                # try:
                #     print(pos_loss_item.shape, neg_loss_item.shape)
                #     loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()
                # except:
                #     import ipdb; ipdb.set_trace()

                
                # for sample_idx_ in range(c_proj_selected.shape[0]):
                #     if sample_idx !=  sample_idx_:
                #         neg_loss_item += torch.exp(F.cosine_similarity(c_proj_selected[sample_idx], c_proj_selected[sample_idx_], dim=-1) / self.tau)
                #         # import ipdb; ipdb.set_trace()
                # print(pos_loss_item, neg_loss_item)

            # nominator = pos_loss_item
            # denominator = neg_loss_item + nominator
            # loss += -torch.log(nominator / (denominator + 1e-8)).mean()
        return loss / batch_size
            # import ipdb; ipdb.set_trace()

        # easy_indices = mask.nonzero()
        # hard_indices = fn_mask.nonzero()

        # num_easy = easy_indices.shape[0]
        # num_hard = hard_indices.shape[0]

        # hard_sample_num = min(sample_num//2, num_hard)
        # easy_sample_num = min(sample_num-hard_sample_num, num_easy)

        # easy_perm = torch.randperm(num_easy)
        # easy_indices = easy_indices[easy_perm[:easy_sample_num]]
        # hard_perm = torch.randperm(num_hard)
        # hard_indices = hard_indices[hard_perm[:hard_sample_num]]
        # indices = torch.cat((hard_indices, easy_indices), dim=0)


        # import ipdb; ipdb.set_trace()

        # labels = labels.contiguous().view(batch_size, -1)
        # predict = predict.contiguous().view(batch_size, -1)
        # feats = feats.permute(0, 2, 3, 1)
        # feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        # feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        # loss = self._contrastive(feats_, labels_)

        # pass
