export CUDA_VISIBLE_DEVICES=3

k=5
# f=50
low=30
high=45
gi=200
cp=ckpts
r=1

nsample=100
ntype=convnet6
gamma=120
phase=0 # end 7
nclass=10
tart_ncls=10
d=6
guide_type=rgd

spec=nette
for f in 17 20 25; do

        python igd_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
        --save-dir "./results/dit-${guide_type}-dino/${spec}-${nsample}-${cp}-${ntype}-k${k}-f${f}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /root/share/ImageNet/train \
        --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth 10 \
        --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --lambda-neg ${gamma} \
        --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls} --guide_type ${guide_type} --f-scale ${f}

        python train.py -d imagenet --imagenet_dir ./results/dit-${guide_type}-dino/${spec}-${nsample}-${cp}-${ntype}-k${k}-f${f}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
                -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag rgd_dino_ipc10_f${f} --slct_type random --spec ${spec} --repeat 3
        
        python train.py -d imagenet --imagenet_dir ./results/dit-${guide_type}-dino/${spec}-${nsample}-${cp}-${ntype}-k${k}-f${f}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
                -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag rgd_dino_ipc50_f${f} --slct_type random --spec ${spec} --repeat 3

        python train.py -d imagenet --imagenet_dir ./results/dit-${guide_type}-dino/${spec}-${nsample}-${cp}-${ntype}-k${k}-f${f}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
                -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 100 --tag rgd_dino_ipc100_f${f} --slct_type random --spec ${spec} --repeat 1

done