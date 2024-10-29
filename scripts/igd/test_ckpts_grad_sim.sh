export CUDA_VISIBLE_DEVICES=3

python test_ckpts_grad_sim.py --data-path /root/share/ImageNet/train \
    --ckpt-path ./ckpts/${spec}/${model}/ \
    --depth 6 --nclass 10 --spec woof 
