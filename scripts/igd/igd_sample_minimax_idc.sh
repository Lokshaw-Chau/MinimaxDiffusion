# train ckpt
model=convnet6
spec=woof

export CUDA_VISIBLE_DEVICES=2

# python train_ckpts.py -d imagenet --imagenet_dir /root/share/ImageNet/train /root/share/ImageNet \
#     -n ${model} --nclass 10 --norm_type instance --tag test --slct_type random --spec ${spec} --mixup vanilla --repeat 1 \
#     --ckpt-dir ./ckpts/${spec}/${model}/ --epochs 50 --depth 10 --ipc -1

spec=idc
python train_ckpts.py -d imagenet --imagenet_dir /root/share/ImageNet/train /root/share/ImageNet \
    -n ${model} --nclass 10 --norm_type instance --tag test --slct_type random --spec ${spec} --mixup vanilla --repeat 1 \
    --ckpt-dir ./ckpts/${spec}/${model}/ --epochs 50 --depth 10 --ipc -1
# influence guided sample

# evaluation