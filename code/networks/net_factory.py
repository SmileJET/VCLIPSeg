# encoding=utf-8

from networks.VNet_clip import VCLIPSeg


def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train", txt_emb=None):
    
    if net_type == "vclipseg" and mode == "train":
        net = VCLIPSeg(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, txt_emb=txt_emb).cuda()
    elif net_type == "vclipseg" and mode == "test":
        net = VCLIPSeg(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, txt_emb=txt_emb).cuda()   
    
    return net
