# VCLIPSeg: Voxel-Wise CLIP-Enhanced Model for Semi-supervised Medical Image Segmentation
by Lei Li, Sheng Lian, Zhiming Luo*, Beizhan Wang, Shaozi Li

## Introduction
This repository is for our paper: '[VCLIPSeg: Voxel-Wise CLIP-Enhanced Model for Semi-supervised Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-72114-4_66)'.

## Usage
1. Clone the repo.;

``` bash
git clone https://github.com/SmileJET/VCLIPSeg.git
```

2. Put the data in './VCLIPSeg/data';

3. Generate the Text embeddings in './VCLIPSeg/clip_embedding';

4. Train & Test the model;

```bash
sh train.sh
```

## Citation

If our VCLIPSeg model is useful for your research, please consider citing:

```
@inproceedings{li2024vclipseg,
  title={VCLIPSeg: Voxel-Wise CLIP-Enhanced Model for Semi-supervised Medical Image Segmentation},
  author={Li, Lei and Lian, Sheng and Luo, Zhiming and Wang, Beizhan and Li, Shaozi},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={692--701},
  year={2024},
  organization={Springer}
}
```

## Acknowledgements
Our code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [MCNet](https://github.com/ycwu1997/MC-Net) and [CLIP-Driven-Universal-Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.