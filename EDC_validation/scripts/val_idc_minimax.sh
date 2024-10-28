# cd /root/workspace/MinimaxDiffusion/

# CUDA_VISIBLE_DEVICES=2 python sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/logs/ImagenetIDC_run-0/000-DiT-XL-2-minimax/checkpoints/0012000.pt \
#         --save-dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-minimax_scaling_law --spec idc --num-samples 1000

cd /root/workspace/MinimaxDiffusion/validation

# CUDA_VISIBLE_DEVICES=2 python /root/workspace/MinimaxDiffusion/validation/main.py \
# --subset "imagenet-10" \
# --spec "idc" \
# --tag "minimax_fkd" \
# --arch-name "resnet18" \
# --factor 2 \
# --mipc 1000 \
# --ipc 1 \
# --stud-name "resnet18" \
# --re-epochs 1000 \
# --re-batch-size 50 \
# --val-dir '/root/share/ImageNet/val' \
# --syn-data-path '/root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-minimax_scaling_law' 

CUDA_VISIBLE_DEVICES=2 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-10" \
--spec "idc" \
--tag "minimax_fkd" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 5 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-minimax_scaling_law' 

CUDA_VISIBLE_DEVICES=2 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-10" \
--spec "idc" \
--tag "minimax_fkd" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 100 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-minimax_scaling_law' 

CUDA_VISIBLE_DEVICES=2 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-10" \
--spec "idc" \
--tag "minimax_fkd" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 500 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-minimax_scaling_law' 

CUDA_VISIBLE_DEVICES=2 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-10" \
--spec "idc" \
--tag "minimax_fkd" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 200 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-minimax_scaling_law' 