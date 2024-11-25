export CUDA_VISIBLE_DEVICES=3

save_path=/root/workspace/MinimaxDiffusion/results/dit-igd/woof-100-dit-igd-ckpts-convnet6-k5-gamma120-r1-gi200-low30-high45
spec=woof
tag=igd



python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag ${tag} --slct_type random --spec ${spec} --repeat 3

# python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag ${tag} --slct_type random --spec ${spec} --repeat 3

# python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 100 --tag ${tag} --slct_type random --spec ${spec} --repeat 1