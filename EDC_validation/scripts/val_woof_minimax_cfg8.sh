cd /root/workspace/MinimaxDiffusion/

CUDA_VISIBLE_DEVICES=0 python sample.py --model DiT-XL/2 --image-size 256  --cfg-scale 8.0 --ckpt /root/workspace/MinimaxDiffusion/logs/run-0/001-DiT-XL-2-minimax/checkpoints/0012000.pt \
        --save-dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-minimax_cfg8_scaling_law --spec woof --num-samples 500

cd /root/workspace/MinimaxDiffusion/validation

CUDA_VISIBLE_DEVICES=0 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "minimax_fkd_cfg_8" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 1 \
--stud-name "resnet18" \
--re-epochs 1000 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-minimax_cfg8_scaling_law' 

CUDA_VISIBLE_DEVICES=0 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "minimax_fkd_cfg_8" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 5 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-minimax_cfg8_scaling_law' 

CUDA_VISIBLE_DEVICES=0 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "minimax_fkd_cfg_8" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 100 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path '/root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-minimax_cfg8_scaling_law'

CUDA_VISIBLE_DEVICES=0 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "minimax_fkd_cfg_8" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 500 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-minimax_cfg8_scaling_law \

CUDA_VISIBLE_DEVICES=0 python /root/workspace/MinimaxDiffusion/validation/main.py \
--subset "imagenet-woof" \
--spec "woof" \
--tag "minimax_fkd_cfg_8" \
--arch-name "resnet18" \
--factor 2 \
--mipc 1000 \
--ipc 200 \
--stud-name "resnet18" \
--re-epochs 1000 \
--re-batch-size 50 \
--val-dir '/root/share/ImageNet/val' \
--syn-data-path /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-minimax_cfg8_scaling_law 