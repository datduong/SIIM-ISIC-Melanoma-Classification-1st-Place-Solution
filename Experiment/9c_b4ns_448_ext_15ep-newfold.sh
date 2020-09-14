#!/bin/bash

# sbatch --partition=gpu --time=2-12:00:00 --gres=gpu:p100:2 --mem=16g --cpus-per-task=24
# sinteractive --partition=gpu --gres=gpu:k80:1 --mem=4g --cpus-per-task=8
# sbatch --partition=gpu --time=2-12:00:00 --gres=gpu:k80:2 --mem=12g --cpus-per-task=24

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
datadir=/data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte
modeldir=/data/duongdb/ISIC2020-SkinCancerBinary/tf_efficientnet_b4_ns
logdir=/data/duongdb/ISIC2020-SkinCancerBinary/tf_efficientnet_b4_ns
oofdir=/data/duongdb/ISIC2020-SkinCancerBinary/tf_efficientnet_b4_ns/EvalDev

python train.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir $datadir --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1 --model-dir $modeldir --log-dir $logdir --num-workers 8 --fold '4'


# python evaluate.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir $datadir --model-dir $modeldir --log-dir $logdir --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 8 --CUDA_VISIBLE_DEVICES 0,1 



