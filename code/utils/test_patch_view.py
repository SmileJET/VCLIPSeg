import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from utils import losses

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    if dataset_name == "LA":
        with open('./data/LA/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    elif dataset_name == "Pancreas_CT":
        with open('./data/Pancreas/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case(model_name, num_outputs, model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0):

    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    dice_loss = losses.Binary_dice_loss
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map, label_map2, score_map2, weight_map, weight_map2 = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # import ipdb; ipdb.set_trace();
        loss1 = dice_loss(torch.Tensor(score_map[1, ...]), torch.Tensor(label==1))
        loss2 = dice_loss(torch.Tensor(score_map2[1, ...]), torch.Tensor(label==1))
        dice_list = [loss1, loss2]
        # V1
        dice_list = np.array(dice_list)
        # sigmoid
        dice_list = 1 / (1 + np.exp(-dice_list))
        dice_list = 1 - dice_list
        # softmax
        pse_weight = np.exp(dice_list)/np.sum(np.exp(dice_list))
        all = np.zeros((2,)+score_map.shape)
        all[0] = pse_weight[0] * score_map
        all[1] = pse_weight[1] * score_map2
        # pseudo_label = all.sum(0).argmax(0)
        pseudo_label = (all.sum(0)[0]>0.5).astype(np.uint8)
        # import ipdb; ipdb.set_trace()
        # prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if num_outputs > 1:
            prediction_average, score_map_average = test_single_case_average_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            if num_outputs > 1:
                 prediction_average = getLargestCC(prediction_average)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
            if num_outputs > 1:
                single_metric_average = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            if num_outputs > 1:
                single_metric_average  = calculate_metric_percase(prediction_average, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
            if num_outputs > 1:
                print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric_average[0], single_metric_average[1], single_metric_average[2], single_metric_average[3]))
        
        total_metric += np.asarray(single_metric)
        if num_outputs > 1:
            total_metric_average += np.asarray(single_metric_average) 
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            if num_outputs > 1:
                nib.save(nib.Nifti1Image(prediction_average.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred_average.nii.gz" % ith)
                nib.save(nib.Nifti1Image(score_map_average[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores_average.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_gt.nii.gz" % ith)


            nib.save(nib.Nifti1Image(weight_map[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_fusion_weight.nii.gz" % ith)
            nib.save(nib.Nifti1Image(weight_map2[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_fusion_weight2.nii.gz" % ith)
            nib.save(nib.Nifti1Image(pseudo_label[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_pseudo_label.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label_map2[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred2.nii.gz" % ith)
        
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is decoder 1 {}'.format(avg_metric))
    if num_outputs > 1:
        avg_metric_average = total_metric_average / len(image_list)
        print('average metric of all decoders is {}'.format(avg_metric_average))
    
    with open(test_save_path+'../{}_performance.txt'.format(model_name), 'w') as f:
        f.writelines('average metric of decoder 1 is {} \n'.format(avg_metric))
        if num_outputs > 1:
            f.writelines('average metric of all decoders is {} \n'.format(avg_metric_average))
    return avg_metric


def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    score_map2 = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    weight_map2 = np.zeros((1, ) + image.shape).astype(np.float32)
    weight_map = np.zeros((1, image.shape[0]//16, image.shape[1]//16, image.shape[2]//16)).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    wcnt = np.zeros((image.shape[0]//16, image.shape[1]//16, image.shape[2]//16)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if isinstance(y, dict):
                        y = y['out']
                    elif len(y) > 1:
                        out_seg1, out_seg2, pred1, pred2, fea1, fea2, fusion_weight = y
                        # y = y[0]
                        y = out_seg1
                        y2 = out_seg2
                        
                        # import ipdb; ipdb.set_trace();
                    y = F.softmax(y, dim=1)
                    y2 = F.softmax(y2, dim=1)
                y = y.cpu().data.numpy()
                y2 = y2.cpu().data.numpy()
                # import ipdb; ipdb.set_trace();
                we = fusion_weight[:, :, :, :, :, 1]
                fu = F.interpolate(we, y.shape[-3:])
                fusion_weight = fusion_weight.cpu().data.numpy()
                y = y[0,1,:,:,:]
                y2 = y2[0,1,:,:,:]
                fu = fu[0,0,:, :, :].cpu().data.numpy()
                fusion_weight = fusion_weight[0, :, :, :, :, 1]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                score_map2[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y2
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                weight_map[:, xs//16:xs//16+patch_size[0]//16, ys//16:ys//16+patch_size[1]//16, zs//16:zs//16+patch_size[2]//16] \
                  = weight_map[:, xs//16:xs//16+patch_size[0]//16, ys//16:ys//16+patch_size[1]//16, zs//16:zs//16+patch_size[2]//16] + fusion_weight
                wcnt[xs//16:xs//16+patch_size[0]//16, ys//16:ys//16+patch_size[1]//16, zs//16:zs//16+patch_size[2]//16] \
                  = wcnt[xs//16:xs//16+patch_size[0]//16, ys//16:ys//16+patch_size[1]//16, zs//16:zs//16+patch_size[2]//16] + 1
                weight_map2[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = weight_map2[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + fu
                # import ipdb; ipdb.set_trace();

    score_map = score_map/np.expand_dims(cnt,axis=0)
    score_map2 = score_map2/np.expand_dims(cnt,axis=0)
    weight_map = weight_map/np.expand_dims(wcnt, axis=0)
    weight_map2 = weight_map2/np.expand_dims(cnt, axis=0)
    label_map = (score_map[0]>0.5).astype(np.uint8)
    label_map2 = (score_map2[0]>0.5).astype(np.uint8)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        label_map2 = label_map2[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map2 = score_map2[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        weight_map = weight_map[:,wl_pad//16:wl_pad//16+w//16,hl_pad//16:hl_pad//16+h//16,dl_pad//16:dl_pad//16+d//16]
        weight_map2 = weight_map2[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map, label_map2, score_map2, weight_map[0], weight_map2[0]

def test_single_case_average_output(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y_logit = net(test_patch)
                    num_outputs = len(y_logit)
                    y=torch.zeros(y_logit[0].shape).cuda()
                    for idx in range(num_outputs):
                        y += y_logit[idx]
                    y/=num_outputs
                    
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice*100, jc*100, hd, asd
