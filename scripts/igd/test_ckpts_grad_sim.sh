export CUDA_VISIBLE_DEVICES=3

spec=nette
model=convnet6

python test_ckpts_grad_sim.py --data-path /root/share/ImageNet/train \
    --ckpt-path ./ckpts \
    --depth 6 --nclass 10 --spec nette 
