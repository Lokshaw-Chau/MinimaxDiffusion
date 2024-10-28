export CUDA_VISIBLE_DEVICES=3

# not RandAug
# python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc10-minimax_baseline /root/share/ImageNet \
#     -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec woof --tag EDC \
#     --epochs 1000 --batch_size 50 --soft_label none --repeat 3

# python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc50-minimax_baseline /root/share/ImageNet \
#     -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec woof --tag EDC \
#     --epochs 1000 --batch_size 50 --soft_label none --repeat 3

python train.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc50-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec none --tag baseline --repeat 3
    # --epochs 1000 --batch_size 50 --soft_label none 

python train.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc10-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec none --tag baseline --repeat 3
    # --epochs 1000 --batch_size 50 --soft_label none 

# python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc10-minimax_baseline /root/share/ImageNet \
#     -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec woof --tag EDC_soft_label \
#     --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3

# python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc50-minimax_baseline /root/share/ImageNet \
#     -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec woof --tag EDC_soft_label \
#     --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3

# python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc50-minimax_baseline /root/share/ImageNet \
#     -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec none --tag EDC_soft_label \
#     --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3


