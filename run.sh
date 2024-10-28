export CUDA_VISIBLE_DEVICES=0

# fine-tune the diffusion model
# torchrun --nnode=1 --master_port=25678 train_dit.py --model DiT-XL/2 \
#      --data-path /root/share/ImageNet/train --ckpt pretrained_models/DiT-XL-2-256x256.pt \
#      --global-batch-size 8 --tag minimax --ckpt-every 12000 --log-every 1500 --epochs 8 \
#      --condense --finetune-ipc -1 --results-dir ./logs/run-0 --spec woof


for i in  {1,2,3,4,5}
do
    # run sample generation
    python sample.py --model DiT-XL/2 --image-size 256 --ckpt ./logs/run-0/001-DiT-XL-2-minimax/checkpoints/0012000.pt \
        --save-dir ./results/dit-distillation/imagenet-10-1000-minimax_$i --spec woof --seed $i

    # # run validation
    python train.py -d imagenet --imagenet_dir ./results/dit-distillation/imagenet-10-1000-minimax_$i /root/share/ImageNet \
        -n resnet_ap --nclass 10 --norm_type instance --ipc 100 --tag test --slct_type random --spec woof --tag restnet10ap_ipc100_$i
done