
# ! let's do a test run to see if we can get the same numbers


python train.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --use-meta --n-epochs 18 --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_b4ns_2e_896_ext_15ep --data-dir ./data/ --data-folder 1024 --image-size 896 --enet-type tf_efficientnet_b4_ns --use-amp --init-lr 2e-5 --CUDA_VISIBLE_DEVICES 0,1,2,3,4,5

# ! b4, no meta data
#!/bin/bash
# sinteractive --partition=gpu --gres=gpu:p100:1 --mem=4g -c4
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
cd /data/duongdb/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
datadir=/data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte
modeldir=/data/duongdb/ISIC2020-SkinCancerBinary/tf_efficientnet_b4_ns
logdir=/data/duongdb/ISIC2020-SkinCancerBinary/tf_efficientnet_b4_ns
python train.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir $datadir --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0 --model-dir $modeldir --log-dir $logdir

python train.py --kernel-type 9c_b4ns_768_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b4ns_768_768_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_meta_b4ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns --use-meta --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 4c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --out-dim 4 --init-lr 1.5e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --init-lr 1.5e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_meta128_32_b5ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b5_ns --use-meta --n-meta-dim 128,32 --use-amp --CUDA_VISIBLE_DEVICES 0

python train.py --kernel-type 9c_b6ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_b6ns_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b6ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b7ns_1e_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b7ns_1e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7

python train.py --kernel-type 9c_meta_1.5e-5_b7ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b7_ns --use-meta --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_nest101_2e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type resnest101 --init-lr 2e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_se_x101_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type seresnext101 --use-amp --CUDA_VISIBLE_DEVICES 0,1
