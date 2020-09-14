#!/bin/bash

# sbatch --partition=gpu --time=2-12:00:00 --gres=gpu:p100:4 --mem=10g --cpus-per-task=20
# sinteractive --partition=gpu --gres=gpu:p100:1 --mem=4g --cpus-per-task=8

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
datadir=/data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte
modeldir=/data/duongdb/ISIC2020-SkinCancerBinary/9c_b7ns_1e_576_ext_15ep_newfold
logdir=/data/duongdb/ISIC2020-SkinCancerBinary/9c_b7ns_1e_576_ext_15ep_newfold

# ! have to reduce batch to use 4 gpu

python train.py --kernel-type 9c_b7ns_1e_576_ext_15ep_newfold --data-dir $datadir --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3 --model-dir $modeldir --log-dir $logdir --num-workers 8 --batch-size 28 --fold '0,1,2'


