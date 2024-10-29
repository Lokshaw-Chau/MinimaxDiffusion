
export CUDA_VISIBLE_DEVICES=3

k=10 # gm-scale
low=30 # guidance interval
high=45
gi=200
cp=ckpts
r=1 # repeat sampling
spec=woof
nsample=100
ntype=convnet6
gamma=100 # weight for diversity constraint
phase=0 # end 7
nclass=10
tart_ncls=10
d=6

python igd_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/logs/run-0/001-DiT-XL-2-minimax/checkpoints/0012000.pt \
    --save-dir "./results/dit-distillation/${spec}-${nsample}-dit-igd-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /root/share/ImageNet/train \
    --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} \
    --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --lambda-neg ${gamma} \
    --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls}

