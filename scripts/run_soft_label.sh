export CUDA_VISIBLE_DEVICES=0
## temp = 1
python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc10-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec woof --tag EDC_soft_label_resnet_temp_1 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 1 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc50-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec woof --tag EDC_soft_label_resnet_temp_1 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 1 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc50-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec none --tag EDC_soft_label_resnet_temp_1 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 1 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc10-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec none --tag EDC_soft_label_resnet_temp_1 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 1 --dsa True

## temp = 3.2
python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc10-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec woof --tag EDC_soft_label_resnet_temp_3.2 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 3.2 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc50-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec woof --tag EDC_soft_label_resnet_temp_3.2 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 3.2 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc50-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec none --tag EDC_soft_label_resnet_temp_3.2 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 3.2 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc10-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec none --tag EDC_soft_label_resnet_temp_3.2 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 3.2 --dsa True

## temp = 5

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc10-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec woof --tag EDC_soft_label_resnet_temp_5 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 5 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc50-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec woof --tag EDC_soft_label_dsa_temp_5 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 5 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc50-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --slct_type random --spec none --tag EDC_soft_label_resnet_temp_5 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 5 --dsa True

python train_edc.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc10-minimax_baseline /root/share/ImageNet \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --slct_type random --spec none --tag EDC_soft_label_resnet_temp_5 \
    --epochs 1000 --batch_size 50 --soft_label vanilla --repeat 3 --temp 5 --dsa True
