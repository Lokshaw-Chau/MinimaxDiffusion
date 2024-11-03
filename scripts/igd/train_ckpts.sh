export CUDA_VISIBLE_DEVICES=2
spec=nette
model=convnet6


python train_ckpts.py -d imagenet --imagenet_dir /root/share/ImageNet/train /root/share/ImageNet \
    -n ${model} --nclass 10 --norm_type instance --tag test --slct_type random --spec ${spec} --mixup vanilla --repeat 1 \
    --ckpt-dir ./ckpts/${spec}/${model}/ --epochs 50 --depth 10 --ipc -1 --lr 0.01