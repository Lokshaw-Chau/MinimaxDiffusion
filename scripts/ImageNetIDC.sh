export CUDA_VISIBLE_DEVICES=0

# fine-tune the diffusion model
# torchrun --nnode=1 --nproc_per_node=1 --master_port=25678 train_dit.py --model DiT-XL/2 \
#      --data-path /root/share/ImageNet/train --ckpt pretrained_models/DiT-XL-2-256x256.pt \
#      --global-batch-size 8 --tag minimax --ckpt-every 12000 --log-every 1500 --epochs 8 \
#      --condense --finetune-ipc -1 --results-dir ./logs/ImagenetIDC_run-0 --spec none --nclass 10


# for i in  {1,2,3,4,5}
# do
#     # run sample generation
#     python sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/logs/ImagenetIDC_run-0/000-DiT-XL-2-minimax/checkpoints/0012000.pt \
#         --save-dir ./results/dit-distillation/ImagenetIDC-ipc10-minimax_$i --spec none --seed $i --num-samples 10

#     # # run validation
#     python train.py -d imagenet --imagenet_dir ./results/dit-distillation/ImagenetIDC-ipc10-minimax_$i /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag test_$i --slct_type random --spec none
# done

for i in  {6,7,8,9,10}
do
    # run sample generation
    python sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/logs/ImagenetIDC_run-0/000-DiT-XL-2-minimax/checkpoints/0012000.pt \
        --save-dir ./results/dit-distillation/ImagenetIDC-ipc10-minimax_$i --spec none --seed $i --num-samples 10

    # run validation
    python train.py -d imagenet --imagenet_dir ./results/dit-distillation/ImagenetIDC-ipc10-minimax_$i /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag IDC_$i --slct_type random --spec none

done