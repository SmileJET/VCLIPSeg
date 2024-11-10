gpuid=0

python ./code/train_3d.py --dataset_name LA --model vclipseg --exp VCLIPSeg --labelnum 4 --gpu $gpuid --temperature 0.1 --sim_w 0.5 --scale_factor 0.0625 && \
python ./code/test_3d.py --dataset_name LA --model vclipseg --exp VCLIPSeg --labelnum 4 --gpu $gpuid  --sim_w 0.5&& \

python ./code/train_3d.py --dataset_name LA --model vclipseg --exp VCLIPSeg --labelnum 8 --gpu $gpuid --temperature 0.1 --sim_w 0.5 --scale_factor 0.0625 && \
python ./code/test_3d.py --dataset_name LA --model vclipseg --exp VCLIPSeg --labelnum 8 --gpu $gpuid  --sim_w 0.5&& \

python ./code/train_3d.py --dataset_name LA --model vclipseg --exp VCLIPSeg --labelnum 16 --gpu $gpuid --temperature 0.1 --sim_w 0.5 --scale_factor 0.0625 && \
python ./code/test_3d.py --dataset_name LA --model vclipseg --exp VCLIPSeg --labelnum 16 --gpu $gpuid  --sim_w 0.5&& \

exit 0;