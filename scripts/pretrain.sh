export CUDA_VISIBLE_DEVICES=0

python train.py -d imagenet --imagenet_dir /root/share/ImageNet \
    -n resnet18 --nclass 10 --norm_type instance --ipc -1 --spec woof --tag pretrain \
    --epochs 20 --batch_size 128 --save_ckpt True

# python train.py -d imagenet --imagenet_dir /root/share/ImageNet \
#     -n mobilenet_v2 --nclass 10 --norm_type instance --ipc -1 --spec woof --tag pretrain \
#     --epochs 20 --batch_size 128 --save_ckpt True

# python train.py -d imagenet --imagenet_dir /root/share/ImageNet \
#     -n efficientnet_b0 --nclass 10 --norm_type instance --ipc -1 --spec woof --tag pretrain \
#     --epochs 20 --batch_size 128 --save_ckpt True

# python train.py -d imagenet --imagenet_dir /root/share/ImageNet \
#     -n shufflenet_v2_x0_5 --nclass 10 --norm_type instance --ipc -1 --spec woof --tag pretrain \
#     --epochs 20 --batch_size 128 --save_ckpt True

python train.py -d imagenet --imagenet_dir /root/share/ImageNet \
    -n resnet18 --nclass 10 --norm_type instance --ipc -1 --spec none --tag pretrain \
    --epochs 20 --batch_size 128 --save_ckpt True

# python train.py -d imagenet --imagenet_dir /root/share/ImageNet \
#     -n mobilenet_v2 --nclass 10 --norm_type instance --ipc -1 --spec none --tag pretrain \
#     --epochs 20 --batch_size 128 --save_ckpt True

# python train.py -d imagenet --imagenet_dir /root/share/ImageNet \
#     -n efficientnet_b0 --nclass 10 --norm_type instance --ipc -1 --spec none --tag pretrain \
#     --epochs 20 --batch_size 128 --save_ckpt True

# python train.py -d imagenet --imagenet_dir /root/share/ImageNet \
#     -n shufflenet_v2_x0_5 --nclass 10 --norm_type instance --ipc -1 --spec none --tag pretrain \
#     --epochs 20 --batch_size 128 --save_ckpt True

    