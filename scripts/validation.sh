export CUDA_VISIBLE_DEVICES=3

# fine-tune the diffusion model
# torchrun --nnode=1 --nproc_per_node=1 --master_port=25678 train_dit.py --model DiT-XL/2 \
#      --data-path /root/share/ImageNet/train --ckpt pretrained_models/DiT-XL-2-256x256.pt \
#      --global-batch-size 32 --tag minimax_ --ckpt-every 12000 --log-every 1500 --epochs 8 \
#      --condense --finetune-ipc -1 --results-dir ./logs/ImagenetIDC_run-00 --spec none --nclass 10


for i in  {1,2,3,4,5}
do

    # run validation
    python train.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc10-minimax_$i /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag idc_$i --slct_type random --spec none

    python train.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc10-minimax_$i /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag woof_$i --slct_type random --spec woof 

done

for i in  {1,2,3,4,5}
do

    # run validation
    python train.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetIDC-ipc50-minimax_$i /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag idc_$i --slct_type random --spec none

    python train.py -d imagenet --imagenet_dir /root/workspace/MinimaxDiffusion/results/dit-distillation/ImagenetWoof-ipc50-minimax_$i /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag woof_$i --slct_type random --spec woof 

done
# for i in  {1,2,3,4,5}
# do
#     # run sample generation
#     python sample.py --model DiT-XL/2 --image-size 256 --ckpt ./logs/run-0/001-DiT-XL-2-minimax/checkpoints/0012000.pt \
#         --save-dir ./results/dit-distillation/imagenet-10-50-minimax_$i --spec woof --seed $i --num-samples 50

#     # # run validation
#     python train.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-10-minimax_$i /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag test_$i --slct_type random --spec woof 
# done