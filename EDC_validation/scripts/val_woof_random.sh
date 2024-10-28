

# CUDA_VISIBLE_DEVICES=5 python /root/workspace/MinimaxDiffusion/validation/main.py \
# --subset "imagenet-woof" \
# --spec "woof" \
# --tag "random_fkd" \
# --arch-name "resnet18" \
# --factor 2 \
# --mipc 1000 \
# --ipc 1 \
# --stud-name "resnet18" \
# --re-epochs 1000 \
# --val-dir '/root/share/ImageNet/val' \
# --syn-data-path '/root/share/ImageNet/train' 

CUDA_VISIBLE_DEVICES=5 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "random_fkd" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 5 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/share/ImageNet/train' 

CUDA_VISIBLE_DEVICES=5 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "random_fkd" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 100 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/share/ImageNet/train' \

CUDA_VISIBLE_DEVICES=5 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "random_fkd" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 500 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/share/ImageNet/train' \

CUDA_VISIBLE_DEVICES=5 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "random_fkd" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 200 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/share/ImageNet/train' 