'''
Date: 2023-10-03 03:14:01
LastEditors: ll
LastEditTime: 2023-11-03 13:08:20
FilePath: /SSL/CLIP_DrivedSSL_LA/code/test_3d_prototype_txt_emb.py
'''
import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MCNet', help='exp_name')
parser.add_argument('--model', type=str,  default='mcnet3d_v1', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--sim_w', type=float, default=0.5, help='weight to balance all losses')


FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
sim_w_txt = ('%0.2f'%(FLAGS.sim_w)).replace('.', '_')
snapshot_path = FLAGS.root_path + "model/{}_{}_{}_labeled_simw_{}/{}".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, sim_w_txt, FLAGS.model)
test_save_path = FLAGS.root_path + "model/{}_{}_{}_labeled_simw_{}/{}_predictions/".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, sim_w_txt, FLAGS.model)

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    FLAGS.root_path = FLAGS.root_path + 'data/LA'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    txt_emb = torch.load('clip_embedding/la_txt_encoding.pth').cuda()

elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'data/Pancreas'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    txt_emb = torch.load('clip_embedding/pancreas_txt_encoding.pth').cuda()

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

def test_calculate_metric():
    
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test", txt_emb=txt_emb)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path), strict=True)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
