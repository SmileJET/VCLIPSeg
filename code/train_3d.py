import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='MCNet', help='exp_name')
parser.add_argument('--model', type=str,  default='mcnet3d_v1', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--sim_w', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--scale_factor', type=float, default=0.5, help='weight to balance all losses')

args = parser.parse_args()

sim_w_txt = ('%0.2f'%(args.sim_w)).replace('.', '_')
snapshot_path = args.root_path + "model/{}_{}_{}_labeled_simw_{}/{}".format(args.dataset_name, args.exp, args.labelnum, sim_w_txt, args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path+'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path+'data/Pancreas'
    args.max_samples = 62
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                        split='train',
                        transform = transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                            ]))
        # 2 * 512
        txt_emb = torch.load('clip_embedding/la_txt_encoding.pth').cuda()
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
        # 2 * 512
        txt_emb = torch.load('clip_embedding/pancreas_txt_encoding.pth').cuda()

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train", txt_emb=txt_emb)
    
    labelnum = args.labelnum  
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = MultiEpochsDataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs = model(volume_batch)
            num_outputs = len(outputs) // 3

            pred1, pred2 = outputs[2][:labeled_bs], outputs[3][:labeled_bs]
            fea1, fea2 = outputs[4][:labeled_bs], outputs[5][:labeled_bs] 

            gt = F.interpolate(label_batch[:labeled_bs].unsqueeze(dim=1).byte(), scale_factor=args.scale_factor, mode='nearest').view(labeled_bs, -1)
            fea_mean = []
            for idx in range(txt_emb.shape[0]):
                fea_mean.append(torch.cat((fea1[(gt==idx)&(pred1==idx)&(pred2==idx)], fea2[(gt==idx)&(pred1==idx)&(pred2==idx)]), dim=0).mean(dim=0).unsqueeze(dim=0))

            sim = F.cosine_similarity(torch.cat(fea_mean), txt_emb.detach()).mean()
            loss_sim = 1 - sim


            y_ori = torch.zeros((num_outputs,) + outputs[0].shape, device=volume_batch.device)
            y_ori_softmax = torch.zeros((num_outputs,)+outputs[0].shape, device=volume_batch.device)

            loss_seg = 0
            loss_seg_dice = 0 
            dice_list = []
            for idx in range(num_outputs):
                y = outputs[idx][:labeled_bs,...]
                y_prob = F.softmax(y, dim=1)
                tmp_dice = dice_loss(y_prob[:,1,...], label_batch[:labeled_bs,...] == 1)
                dice_list.append(tmp_dice.item())
                loss_seg_dice += tmp_dice
                
                y_all = outputs[idx]
                y_ori[idx] = y_all
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori_softmax[idx] = y_prob_all

            
            dice_list = np.array(dice_list)
            # sigmoid
            dice_list = 1 / (1 + np.exp(-dice_list))
            dice_list = 1 - dice_list
            # softmax
            pse_weight = np.exp(dice_list)/np.sum(np.exp(dice_list))

            all = torch.zeros_like(y_ori_softmax).cuda()
            for i in range(len(y_ori_softmax)):
                all[i] = y_ori_softmax[i] * pse_weight[i]
            y_pseudo_label = all.sum(dim=0).argmax(dim=1)


            loss_consist = 0
            for i in range(num_outputs):
                loss_consist += F.cross_entropy(y_ori[i], y_pseudo_label.long())
                loss_consist += dice_loss(y_ori_softmax[i][:, 1, ...], y_pseudo_label==1)
            
            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            loss = args.lamda * loss_seg_dice + consistency_weight * loss_consist + args.sim_w * loss_sim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print()
            
            logging.info('Exp: %s LabelNum: %d Model: %s iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f, loss_sim: %03f' % (args.exp, args.labelnum, args.model, iter_num, loss, loss_seg_dice, loss_consist, loss_sim))
                
            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name =="LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                elif args.dataset_name =="Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = 'Pancreas_CT')
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)
                model.train()

                logging.info('Curr Val Dice: %.4f, Best Val Dice: %.4f' % (dice_sample, best_dice))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
