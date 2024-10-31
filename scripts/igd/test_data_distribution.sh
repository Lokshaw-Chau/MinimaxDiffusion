export CUDA_VISIBLE_DEVICES=0

spec=woof
model=convnet6

python /root/workspace/MinimaxDiffusion/exam_data_distribution.py -d imagenet --imagenet_dir /root/share/ImageNet/val /root/workspace/MinimaxDiffusion/results/dit-rgd/woof-100-ckpts-convnet6-k5-f50-gamma120-r1-gi200-low30-high45 \
    -n ${model} --nclass 10 --norm_type instance --tag test --slct_type random --spec ${spec} --mixup vanilla --repeat 1 \
    --ckpt-dir ./ckpts/${spec}/${model}/ --epochs 50 --depth 10 --ipc -1 --lr 0.01